import pyrealsense2 as rs
import numpy as np 
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import socket # UDP 통신을 위함

class RealSenseLocalizationStreamer:
    """
    - 실시간 스트리밍, 객체 감지/추적, 자세 추정, 카메라 위치 추정 기능 통합
    """

    TARGET_CLASSES = {'dining table', 'cookie', 'person'} # 탐지 목표 클래스
    MIN_CONF_DET = 0.40
    MIN_CONF_POSE = 0.40
    MAX_DEPTH_M = 5.0
    DEPTH_ALPHA = 0.08

    COCO_SKELETON_BODY = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    HEAD_ID, L_SH_ID, R_SH_ID = 0, 5, 6

    def __init__(self,
                 yolo_det_weights = 'yolov8n.pt',
                 yolo_pose_weights = 'yolov8n-pose.pt',
                 tracker_max_age = 5,
                 show_windows=True,
                 on_detections=None,
                 on_poses=None,
                 on_localizaion=None):
        
        self.det_model = YOLO(yolo_det_weights)
        self.pose_model = YOLO(yolo_pose_weights)
        self.tracker = DeepSort(max_age = tracker_max_age)
        self.on_detections = on_detections
        self.on_poses = on_poses
        self.show_windows = show_windows
        self.on_localization = on_localizaion
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        self.marker_length_m = 0.20
        self.marker_ids = set(range(12))

        # ✅ 실제 맵 (BEV_guide 기반, m 단위, 원점 = ID0)
        self.marker_world_pos = {
            0:  np.array([0.0, 0.0, 0.0], dtype=np.float32),
            1:  np.array([1.5, 0.0, 0.0], dtype=np.float32),
            2:  np.array([3.2, 0.0, 0.0], dtype=np.float32),
            3:  np.array([5.0, 0.0, 0.0], dtype=np.float32),

            4:  np.array([0.0, 1.5, 0.0], dtype=np.float32),
            5:  np.array([1.6, 1.5, 0.0], dtype=np.float32),
            6:  np.array([3.2, 1.5, 0.0], dtype=np.float32),
            7:  np.array([4.8, 1.5, 0.0], dtype=np.float32),

            8:  np.array([0.0, 3.0, 0.0], dtype=np.float32),
            9:  np.array([1.5, 3.0, 0.0], dtype=np.float32),
            10: np.array([3.1, 3.0, 0.0], dtype=np.float32),
            11: np.array([5.0, 3.0, 0.0], dtype=np.float32),
        }
        
        self.axis_scale = 0.15
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
        self.dist = np.array(intr.coeffs, dtype=np.float32)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('localhost', 12345) #데이터를 보낼 주소

        self.target_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_server_address = ('localhost', 12346)   # 위 주소는 카메라 좌표 용, 타겟 좌표용 주소 생성
        
    def _read_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        return np.asanyarray(depth_frame.get_data()), np.asanyarray(color_frame.get_data())
    
    def _depth_at(self, depth_image, cx, cy):
        if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
            return float(depth_image[cy, cx]) * float(self.depth_scale)
        return None
    
    def _safe_int_box(self, box_xyxy):
        return map(int, box_xyxy.tolist() if hasattr(box_xyxy, 'tolist') else box_xyxy)

    def _draw_pose_with_headrule(self, img, kpts_xy):
        for idx, (x, y) in enumerate(kpts_xy):
            if x is None or y is None: continue
            if idx == 0: cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
            elif idx in (1, 2, 3, 4): continue
            else: cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
        for a, b in self.COCO_SKELETON_BODY:
            if a < len(kpts_xy) and b < len(kpts_xy):
                xa, ya, xb, yb = *kpts_xy[a], *kpts_xy[b]
                if None not in (xa, ya, xb, yb):
                    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)
        head, lsh, rsh = (kpts_xy[i] if len(kpts_xy) > i else (None, None) for i in [0, 5, 6])
        if None not in head:
            hx, hy = int(head[0]), int(head[1])
            neck = None
            if None not in lsh and None not in rsh: neck = (int((lsh[0] + rsh[0]) / 2), int((lsh[1] + rsh[1]) / 2))
            elif None not in lsh: neck = (int(lsh[0]), int(lsh[1]))
            elif None not in rsh: neck = (int(rsh[0]), int(rsh[1]))
            if neck: cv2.line(img, (hx, hy), neck, (0, 200, 255), 2)

    def _draw_axes_custom(self, img, rvec, tvec, K, dist, scale):
        obj = np.float32([[0,0,0], [scale,0,0], [0,scale,0], [0,0,scale]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        o, x, y, z = [tuple(pt.ravel().astype(int)) for pt in imgpts]
        cv2.line(img, o, x, (0, 0, 255), 3)
        cv2.line(img, o, y, (255, 0, 0), 3)
        cv2.line(img, o, z, (0, 255, 0), 3)

    def _is_rotation_matrix(self, R):
        return np.linalg.norm(np.identity(3) - np.dot(R.T, R)) < 1e-6

    def _rotation_matrix_to_euler_angles(self, R):
        if not self._is_rotation_matrix(R): return np.array([0,0,0])
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def _put_localization_text(self, img, pos, angles_deg):
        roll, pitch, yaw = angles_deg
        cv2.putText(img, f'Cam Pos [m]: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f'Cam Ang [deg]: Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}', (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _put_marker_info_text(self, img, mid, tvec, anchor_xy):
        tx, ty, tz = tvec.flatten()
        dist = np.linalg.norm(tvec)
        text_anchor = (anchor_xy[0] - 80, anchor_xy[1] - 30)
        cv2.putText(img, f'ID: {int(mid)} | Dist: {dist:.2f}m', text_anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(img, f'Pos(cam): x={tx:.2f} y={ty:.2f} z={tz:.2f}', (text_anchor[0], text_anchor[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
    def run(self):
        try:
            while True:
                depth_image, color_image = self._read_frames()
                if depth_image is None: continue

                # --- 객체 탐지, 추적, 포즈는 기존 코드 유지 ---
                det_results = self.det_model(color_image, verbose=False)
                tracker_bboxes = []
                valid_dets = []
                
                for r in det_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = self._safe_int_box(box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        cls_id = int(box.cls[0])
                        cls_name = self.det_model.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])

                        if cls_name in self.TARGET_CLASSES and conf >= self.MIN_CONF_DET:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            d_m = self._depth_at(depth_image, cx, cy)
                            if d_m is not None and 0.0 <= d_m <= self.MAX_DEPTH_M:
                                valid_dets.append((cls_name, d_m, (x1, y1, x2, y2), (cx, cy), conf))
                            
                                # --- 타겟 좌표 UDP 송신 ---
                                # 카메라 pos (이미 위에서 구한 값) + depth를 이용해 타겟 좌표 근사
                                # --- 타겟 좌표 UDP 송신 (실제 월드 좌표) ---
                                if 'pos' in locals() and 'R' in locals() and 'tvec_g' in locals():
                                    # 1) 픽셀 좌표 + 깊이 → 카메라 좌표계
                                    fx, fy = self.K[0, 0], self.K[1, 1]
                                    cx0, cy0 = self.K[0, 2], self.K[1, 2]
                                    Xc = (cx - cx0) * d_m / fx
                                    Yc = (cy - cy0) * d_m / fy
                                    Zc = d_m
                                    Pc = np.array([Xc, Yc, Zc])

                                    # 2) 카메라 → 월드 좌표 변환
                                    Pw = R.T @ (Pc - tvec_g.flatten())

                                    # 3) UDP 송신
                                    coord_str = f"{Pw[0]},{Pw[1]},{Pw[2]}"
                                    self.target_sock.sendto(coord_str.encode('utf-8'),
                                                            self.target_server_address)

                                    print(f"[Target World Coord] {Pw}")




                        if conf >= self.MIN_CONF_DET:
                            tracker_bboxes.append(([x1, y1, w, h], conf, cls_id))
                            if cls_name.lower() != 'person':
                                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            d_vis = self._depth_at(depth_image, cx, cy)
                            
                            label = f'{cls_name} {conf:.2f}'
                            if d_vis and d_vis < self.MAX_DEPTH_M:
                                label += f' | {d_vis:.2f}m'
                            
                            cv2.putText(color_image, label, (x1, max(0, y1 - 6)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                tracks = self.tracker.update_tracks(tracker_bboxes, frame=color_image)
                for t in tracks:
                    if not t.is_confirmed(): continue
                    l, _, r, b = t.to_ltrb()
                    cv2.putText(color_image, f'ID:{str(t.track_id)}', (int(l), max(0, int(b) - 8)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if self.on_detections is not None and valid_dets:
                    self.on_detections(valid_dets)

                # --- ArUco 마커 기반 Localization ---
                corners, ids, _ = self.aruco_detector.detectMarkers(color_image)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_length_m, self.K, self.dist)
                    
                    for i in range(len(ids)):
                        self._draw_axes_custom(color_image, rvecs[i], tvecs[i],
                                               self.K, self.dist, self.axis_scale * 0.6)
                        cx, cy = np.mean(corners[i].reshape(4, 2), axis=0, dtype=int)
                        self._put_marker_info_text(color_image, ids[i][0], tvecs[i], (cx, cy))

                    obj_points, img_points = [], []
                    for i, mid in enumerate(ids.flatten()):
                        if mid in self.marker_ids:
                            half_len = self.marker_length_m / 2.0
                            world_center = self.marker_world_pos[mid]  # ✅ 실제 맵 좌표 사용
                            obj_pts_marker = np.float32([
                                [-half_len,  half_len, 0],
                                [ half_len,  half_len, 0],
                                [ half_len, -half_len, 0],
                                [-half_len, -half_len, 0]
                            ])
                            obj_points.extend(world_center + obj_pts_marker)
                            img_points.extend(corners[i].reshape(4, 2))
                    
                    if len(obj_points) >= 4:
                        ok, rvec_g, tvec_g = cv2.solvePnP(
                            np.array(obj_points), np.array(img_points), self.K, self.dist)
                        if ok:
                            R, _ = cv2.Rodrigues(rvec_g)
                            pos = (-R.T @ tvec_g).flatten()
                            ang = np.degrees(self._rotation_matrix_to_euler_angles(R.T))
                            self._put_localization_text(color_image, pos, ang)

                            coord_str = f"{pos[0]},{pos[1]},{pos[2]},{ang[0]},{ang[1]},{ang[2]}"
                            self.sock.sendto(coord_str.encode('utf-8'), self.server_address)

                if self.show_windows:
                    cv2.imshow('RealSense Integrated Stream', color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.pipeline.stop()
            self.sock.close()
            if self.show_windows:
                cv2.destroyAllWindows()

# 직접 파일을 실행할때는 아래 코드 주석 해제후 실행
# if __name__ == "__main__":
#     streamer = RealSenseLocalizationStreamer(show_windows=True)
#     streamer.run()