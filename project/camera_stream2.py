
import pyrealsense2 as rs
import numpy as np 
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import socket # UDP 통신을 위함
import time
import rospy
from gesture_detector import GestureDetector


class RealSenseLocalizationStreamer:
    """
    - 실시간 스트리밍, 객체 감지/추적, 자세 추정, 카메라 위치 추정 기능 통합
    """

    TARGET_CLASSES = ['chair', 'mouse', 'remote'] # 탐지 목표 클래스
    IGNORE_CLASSES = ['refrigerator', 'tv','laptop', 'kite', 'frisbee', 'airplane', 'bird', 'sports ball'] # 탐지해도 무시할 클래스
    MIN_CONF_DET = 0.40
    MIN_CONF_POSE = 0.40
    MAX_DEPTH_M = 10.0
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
                 publish_point=None):


        self.gesture = GestureDetector()
        self.det_model = YOLO(yolo_det_weights)
        self.pose_model = YOLO(yolo_pose_weights)
        self.tracker = DeepSort(max_age = tracker_max_age)
        self.on_detections = on_detections
        self.on_poses = on_poses
        self.show_windows = show_windows
        self.publish_point = publish_point
        
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
        self.marker_ids = set(range(16))

        # 실제 맵 (BEV_guide 기반, m 단위, 원점 = ID0)
        self.marker_world_pos = {
            0:  np.array([0.0, 0.0, 2.0], dtype=np.float32),
            1:  np.array([1.5, 0.0, 2.0], dtype=np.float32),
            2:  np.array([3.2, 0.0, 2.0], dtype=np.float32),
            3:  np.array([5.0, 0.0, 2.0], dtype=np.float32),

            4:  np.array([0.0, 1.5, 2.0], dtype=np.float32),
            5:  np.array([1.6, 1.5, 2.0], dtype=np.float32),
            6:  np.array([3.2, 1.5, 2.0], dtype=np.float32),
            7:  np.array([4.8, 1.5, 2.0], dtype=np.float32),

            8:  np.array([0.0, 3.0, 2.0], dtype=np.float32),
            9:  np.array([1.5, 3.0, 2.0], dtype=np.float32),
            10: np.array([3.1, 3.0, 2.0], dtype=np.float32),
            11: np.array([5.0, 3.0, 2.0], dtype=np.float32),

            12: np.array([0.75, -0.1, 0.5], dtype=np.float32),
            13: np.array([3.2, -0.1, 0.5], dtype=np.float32),
            14: np.array([0.0, 3.5, 0.5], dtype=np.float32),
            15: np.array([3.2, 3.5, 0.5], dtype=np.float32),

        }
        
        self.axis_scale = 0.15
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.aruco_params.adaptiveThreshWinSizeMin = 5
        self.aruco_params.adaptiveThreshWinSizeMax = 7
        self.aruco_params.adaptiveThreshWinSizeStep = 2
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
        self.dist = np.array(intr.coeffs, dtype=np.float32)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('localhost', 12345) #데이터를 보낼 주소

        self.target_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_server_address = ('localhost', 12346)   # 위 주소는 카메라 좌표 용, 타겟 좌표용 주소 생성

        self.current_position = None
        self.publish_timer = rospy.Timer(rospy.Duration(2.0), self._publish_callback) # 현재 위치, t 초마다 publish 자동 실행
    
    def _publish_callback(self, event):
        '''t초마다 publish'''
        if self.current_position is not None and self.publish_point is not None:
            self.publish_point(self.current_position, "current")

        
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
        cv2.putText(img, f'Cam Pos [m]: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}', (10, 20),
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
                det_positions = []
                
                for r in det_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = self._safe_int_box(box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        cls_id = int(box.cls[0])
                        cls_name = self.det_model.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])

                        # 무시 클래스 필터링
                        if cls_name in self.IGNORE_CLASSES:
                            continue  # 아래 전체(UDP, publish, draw, tracking) 모두 스킵

                        # --- 타겟 객체 검출 및 UDP publish ---
                        if cls_name in self.TARGET_CLASSES and conf >= self.MIN_CONF_DET:
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            d_m = self._depth_at(depth_image, cx, cy)
                            if d_m is not None and 0.0 <= d_m <= self.MAX_DEPTH_M:
                                valid_dets.append((cls_name, d_m, (x1, y1, x2, y2), (cx, cy), conf))
                                
                                # --- 타겟 좌표 UDP 송신 (실제 월드 좌표) ---
                                if hasattr(self, 'current_position') and self.current_position is not None:
                                    fx, fy = self.K[0, 0], self.K[1, 1]
                                    cx0, cy0 = self.K[0, 2], self.K[1, 2]
                                    Xc = (cx - cx0) * d_m / fx
                                    Yc = (cy - cy0) * d_m / fy
                                    Zc = d_m
                                    Pc = np.array([Xc, Yc, Zc])

                                    Pw = R.T @ (Pc - tvec_g.flatten())

                                    coord_str = f"{cls_name},{Pw[0]},{Pw[1]},{Pw[2]}"
                                    self.target_sock.sendto(coord_str.encode('utf-8'),
                                                            self.target_server_address)
                                    det_positions.append(Pw)

                                    print(f"[Target World Coord] {Pw}")
                                    self.publish_point(Pw, "target")

                        # --- Tracking 및 시각화 (무시 클래스는 이미 continue로 스킵됨) ---
                        if conf >= self.MIN_CONF_DET:
                            tracker_bboxes.append(([x1, y1, w, h], conf, cls_id))
                            
                            # if cls_name.lower() != 'person':
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)

                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            d_vis = self._depth_at(depth_image, cx, cy)

                            label = f'{cls_name} {conf:.2f}'
                            if d_vis and d_vis < self.MAX_DEPTH_M:
                                label += f' | {d_vis:.2f}m'

                            cv2.putText(color_image, label, (x1, max(0, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                            # --- Human pose estimation + hand raise detection ---
                            
                        if cls_name.lower() == "person" and conf > self.MIN_CONF_DET:
                            crop_img = color_image[y1:y2, x1:x2]
                            pose_results = self.pose_model(crop_img, verbose=False)

                            for pose in pose_results:
                                if not hasattr(pose, "keypoints") or pose.keypoints is None:
                                    continue

                                # YOLO keypoints: (17, 3)
                                kpts = pose.keypoints.data.cpu().numpy()[0]

                                # Crop → 원본 이미지 절대 좌표
                                abs_kpts = self.gesture.extract_absolute_keypoints(kpts, x1, y1)

                                # 포즈 그리기
                                self._draw_pose_with_headrule(color_image, abs_kpts)

                                # 손 들었는지 체크
                                if self.gesture.is_hand_raised(abs_kpts):
                                    print("🔵 Hand Raised Detected!")
                            
                        # Human pose estimation
                        # if cls_name.lower() == "person" and conf > self.MIN_CONF_DET:
                        #     crop_img = color_image[y1:y2, x1:x2]
                        #     pose_results = self.pose_model(crop_img, verbose=False)
                        #     for pose in pose_results:
                        #         if not hasattr(pose, "keypoints") or pose.keypoints is None:
                        #             continue
                        #         keypoints = pose.keypoints.data.cpu().numpy()
                        #         for person_kpt in keypoints:
                        #             abs_kpts = []
                        #             for (x, y, c) in person_kpt:
                        #                 if np.isnan(x) or np.isnan(y) or c < self.MIN_CONF_POSE:
                        #                     abs_kpts.append((None, None))
                        #                 else:
                        #                     abs_kpts.append((int(x1 + x), int(y1 + y)))
                        #             self._draw_pose_with_headrule(color_image, abs_kpts)

                # --- 객체 추적 ID 표시 ---
                tracks = self.tracker.update_tracks(tracker_bboxes, frame=color_image)
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    l, _, r, b = t.to_ltrb()
                    cv2.putText(color_image, f'ID:{str(t.track_id)}',
                                (int(l), max(0, int(b) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                
                # if self.on_detections is not None and valid_dets:
                #     self.on_detections(valid_dets)

                # --- ArUco 마커 기반 Localization ---
                corners, ids, _ = self.aruco_detector.detectMarkers(color_image)
                if ids is not None and len(ids) > 0:
                    # 각 마커 pose 추정
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_length_m, self.K, self.dist)

                    # 각 마커의 3D 거리(norm) 계산
                    distances = np.linalg.norm(tvecs.reshape(-1, 3), axis=1)
                    valid_mask = distances <= 5.0 # 5m 밖의 마커는 인식하지 않도록 함 (localization 안정성)
                    if not np.any(valid_mask):
                        continue
                    tvecs = tvecs[valid_mask]
                    rvecs = rvecs[valid_mask]
                    corners = [corners[i] for i, v in enumerate(valid_mask) if v]
                    ids = ids[valid_mask]
                    distances = distances[valid_mask]

                    min_idx = np.argmin(distances)
                    # 가장 가까운 마커 선택
                    closest_id = int(ids[min_idx][0])
                    closest_rvec = rvecs[min_idx]
                    closest_tvec = tvecs[min_idx]
                    closest_corners = corners[min_idx]

                    # 선택된 마커만 강조 표시 (녹색)
                    cv2.aruco.drawDetectedMarkers(color_image, [closest_corners],
                                                np.array([[closest_id]]), (0, 255, 0))

                    # 좌표축과 텍스트 표시
                    self._draw_axes_custom(color_image, closest_rvec, closest_tvec,
                                        self.K, self.dist, self.axis_scale * 0.6)
                    cx, cy = np.mean(closest_corners.reshape(4, 2), axis=0, dtype=int)
                    self._put_marker_info_text(color_image, closest_id, closest_tvec, (cx, cy))

                    cv2.setRNGSeed(42)

                    # solvePnPRansac을 위한 3D-2D 대응점 생성
                    if closest_id in self.marker_ids:
                        half_len = self.marker_length_m / 2.0
                        world_center = self.marker_world_pos[closest_id]

                        # 마커 방향 구분 (14,15만 반대 벽)
                        if closest_id in (14, 15):  # 뒤쪽 벽
                            obj_pts_marker = np.float32([
                                [0,  half_len,  half_len],
                                [0, -half_len,  half_len],
                                [0, -half_len, -half_len],
                                [0,  half_len, -half_len]
                            ])
                        else:  # 나머지 마커 (천장/앞벽)
                            obj_pts_marker = np.float32([
                                [-half_len,  half_len, 0],
                                [ half_len,  half_len, 0],
                                [ half_len, -half_len, 0],
                                [-half_len, -half_len, 0]
                            ])
                        obj_points = world_center + obj_pts_marker
                        img_points = closest_corners.reshape(4, 2)



                        

                        # pose 계산 (RANSAC)
                        ok, rvec_g, tvec_g, _ = cv2.solvePnPRansac(
                            np.array(obj_points),
                            np.array(img_points),
                            self.K,
                            self.dist,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                            reprojectionError=3.0,
                            confidence=0.99,
                            iterationsCount=100
                        )

                        if ok:
                            R, _ = cv2.Rodrigues(rvec_g)
                            pos = (-R.T @ tvec_g).flatten()
                            ang = np.degrees(self._rotation_matrix_to_euler_angles(R.T))
                            self._put_localization_text(color_image, pos, ang)

                            # Temporal Filtering (지수 이동 평균)
                            if not hasattr(self, 'filtered_pos') or self.filtered_pos is None:
                                self.filtered_pos = pos
                            else:
                                alpha = 0.8  # 안정성 우선
                                self.filtered_pos = alpha * self.filtered_pos + (1 - alpha) * pos
                            stable_pos = self.filtered_pos

                            # 좌표 publish
                            coord_str = f"{stable_pos[0]},{stable_pos[1]},{stable_pos[2]},{ang[0]},{ang[1]},{ang[2]}"
                            self.sock.sendto(coord_str.encode('utf-8'), self.server_address)

                            if self.publish_point is not None:
                                self.publish_point(stable_pos, "current")

                            self.current_position = stable_pos

                if self.show_windows:
                    cv2.imshow('RealSense Integrated Stream', color_image)
                    cv2.moveWindow('RealSense Integrated Stream', 400, 400) # 좌측 중간에 팝업
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
