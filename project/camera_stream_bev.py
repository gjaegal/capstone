#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import socket
import time
import rospy
from std_msgs.msg import Float32


class RealSenseLocalizationStreamer:

    TARGET_CLASSES = ['chair', 'mouse', 'remote']
    IGNORE_CLASSES = ['refrigerator','tv','laptop','kite','frisbee','airplane','bird','sports ball']
    MIN_CONF_DET = 0.40
    MAX_DEPTH_M = 10.0

    def __init__(self,
                 yolo_det_weights='yolov8n.pt',
                 yolo_pose_weights='yolov8n-pose.pt',
                 tracker_max_age=5,
                 show_windows=True,
                 show_bev=True,
                 publish_point=None):

        # YOLO models
        self.det_model = YOLO(yolo_det_weights)
        self.pose_model = YOLO(yolo_pose_weights)
        self.tracker = DeepSort(max_age=tracker_max_age)

        self.show_windows = show_windows
        self.show_bev = show_bev
        self.publish_point = publish_point

        rospy.Subscriber("/current_yaw_deg", Float32, self.yaw_callback)
        self.offset = 0.0

        # ---------------------------------------
        # RealSense initialization
        # ---------------------------------------
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

        # ---------------------------------------
        # Floor Marker Map (Top-Down)
        # ---------------------------------------
        self.marker_length_m = 0.20
        self.marker_ids = set(range(12))

        self.marker_world_pos = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([2.0, 0.0, 0.0]),
            2: np.array([4.0, 0.0, 0.0]),
            3: np.array([6.0, 0.0, 0.0]),

            4: np.array([0.0, 1.5, 0.0]),
            5: np.array([2.0, 1.5, 0.0]),
            6: np.array([4.0, 1.5, 0.0]),
            7: np.array([6.0, 1.5, 0.0]),

            8: np.array([0.0, 3.0, 0.0]),
            9: np.array([2.0, 3.0, 0.0]),
            10: np.array([4.0, 3.0, 0.0]),
            11: np.array([6.0, 3.0, 0.0]),
        }

        # ---------------------------------------
        # ArUco detector
        # ---------------------------------------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Camera Intrinsics
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=np.float32)
        self.dist = np.array(intr.coeffs, dtype=np.float32)

        # ---------------------------------------
        # UDP sockets
        # ---------------------------------------
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = ('localhost', 12345)

        self.target_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_addr = ('localhost', 12346)

        # last pose for target transform
        self.last_R_world = None
        self.last_t_world = None

        self.current_position = None
        self.publish_timer = rospy.Timer(rospy.Duration(1.5), self._publish_cb)

        # Bird's Eye View parameters
        self.map_width = 800
        self.map_height = 600
        self.meters_per_pixel = 100  # 100 pixels per meter
        self.map_origin_x = 50  # offset from left edge
        self.map_origin_y = 550  # offset from bottom edge


    # =====================================================
    # Utilities
    # =====================================================

    def yaw_callback(self, msg):
        self.offset = msg.data

    def _publish_cb(self, event):
        if self.current_position and self.publish_point:
            self.publish_point(self.current_position, "current")

    def _read_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        d = frames.get_depth_frame()
        c = frames.get_color_frame()
        if not d or not c:
            return None, None
        return np.asanyarray(d.get_data()), np.asanyarray(c.get_data())

    def _depth(self, depth_img, cx, cy):
        if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
            return depth_img[cy, cx] * self.depth_scale
        return None

    # --------------------- Draw axes ---------------------
    def _draw_axes_custom(self, img, rvec, tvec, K, dist, scale):
        obj = np.float32([
            [0,0,0],
            [scale,0,0],
            [0,scale,0],
            [0,0,scale]
        ])
        imgpts,_ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        o,x,y,z = [tuple(pt.ravel().astype(int)) for pt in imgpts]

        cv2.line(img, o, x, (0,0,255), 3)
        cv2.line(img, o, y, (255,0,0), 3)
        cv2.line(img, o, z, (0,255,0), 3)

    def _rotation_matrix_to_euler(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy >= 1e-6:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x,y,z])
    
    # --------------------- Bird's Eye View Visualization ---------------------
    def _world_to_map(self, x, y):
        """Convert world coordinates (meters) to map pixel coordinates"""
        px = int(self.map_origin_x + x * self.meters_per_pixel)
        py = int(self.map_origin_y - y * self.meters_per_pixel)  # flip Y axis
        return px, py
    
    def _create_birdseye_map(self, cam_x, cam_y, cam_yaw):
        """Create bird's eye view map with camera position and orientation"""
        # Create blank map
        map_img = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 240
        
        # Draw grid
        for i in range(0, 8):
            px, py = self._world_to_map(i, 0)
            cv2.line(map_img, (px, 0), (px, self.map_height), (200, 200, 200), 1)
        for j in range(0, 5):
            px, py = self._world_to_map(0, j)
            cv2.line(map_img, (0, py), (self.map_width, py), (200, 200, 200), 1)
        
        # Draw ArUco marker positions
        for marker_id, pos in self.marker_world_pos.items():
            px, py = self._world_to_map(pos[0], pos[1])
            cv2.circle(map_img, (px, py), 5, (100, 100, 255), -1)
            cv2.putText(map_img, str(marker_id), (px + 8, py + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
        # Draw camera position
        cam_px, cam_py = self._world_to_map(cam_x, cam_y)
        
        # Draw camera circle
        cv2.circle(map_img, (cam_px, cam_py), 12, (0, 255, 0), -1)
        cv2.circle(map_img, (cam_px, cam_py), 12, (0, 200, 0), 2)
        
        # Draw direction arrow based on yaw
        arrow_length = 30
        arrow_end_x = int(cam_px + arrow_length * math.cos(math.radians(cam_yaw)))
        arrow_end_y = int(cam_py - arrow_length * math.sin(math.radians(cam_yaw)))  # flip Y
        cv2.arrowedLine(map_img, (cam_px, cam_py), (arrow_end_x, arrow_end_y),
                       (0, 0, 255), 3, tipLength=0.3)
        
        # Draw coordinate info
        cv2.putText(map_img, f"Position: ({cam_x:.2f}, {cam_y:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(map_img, f"Yaw: {cam_yaw:.1f} deg",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw legend
        cv2.putText(map_img, "Legend:", (10, self.map_height - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.circle(map_img, (20, self.map_height - 55), 5, (100, 100, 255), -1)
        cv2.putText(map_img, "ArUco Markers", (35, self.map_height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.circle(map_img, (20, self.map_height - 30), 8, (0, 255, 0), -1)
        cv2.putText(map_img, "Camera", (35, self.map_height - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.arrowedLine(map_img, (15, self.map_height - 10), (30, self.map_height - 10),
                       (0, 0, 255), 2, tipLength=0.4)
        cv2.putText(map_img, "Direction", (35, self.map_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return map_img


    # =====================================================
    # Main run loop
    # =====================================================
    def run(self):
        try:
            while True:

                depth_img, color = self._read_frames()
                if depth_img is None:
                    continue

                # =====================================================
                # ArUco detection
                # =====================================================
                corners, ids, _ = self.aruco_detector.detectMarkers(color)

                if ids is not None and len(ids) > 0:

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_length_m, self.K, self.dist
                    )

                    dist = np.linalg.norm(tvecs.reshape(-1,3), axis=1)
                    valid = dist <= 4.0

                    if np.any(valid):

                        ids     = ids[valid]
                        corners = [corners[i] for i,v in enumerate(valid) if v]
                        rvecs   = rvecs[valid]
                        tvecs   = tvecs[valid]

                        cam_positions = []
                        cam_rotations = []

                        # =====================================================
                        # Multi-marker solvePnP
                        # =====================================================
                        for i, mid_raw in enumerate(ids):

                            mid = int(mid_raw[0])
                            c = corners[i]
                            rvec = rvecs[i]
                            tvec = tvecs[i]

                            # ArUco box
                            cv2.aruco.drawDetectedMarkers(color, [c], np.array([[mid]]), (0,255,0))
                            self._draw_axes_custom(color, rvec, tvec, self.K, self.dist, self.marker_length_m*0.6)

                            cx, cy = np.mean(c.reshape(4,2), axis=0, dtype=int)

                            # ===== World coord print =====
                            wc = self.marker_world_pos[mid]
                            cv2.putText(color,
                                        f"ID:{mid}  Cam=({tvec[0][0]:.2f},{tvec[0][1]:.2f},{tvec[0][2]:.2f})",
                                        (cx+10, cy+20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45, (0,255,255), 2)

                            # solvePnP for world
                            half = self.marker_length_m / 2
                            world_center = self.marker_world_pos[mid]

                            obj_pts = np.float32([
                                [-half, +half, 0],
                                [+half, +half, 0],
                                [+half, -half, 0],
                                [-half, -half, 0],
                            ])
                            obj_abs = world_center + obj_pts
                            img_pts = c.reshape(4,2)

                            ok, rvec_g, tvec_g, _ = cv2.solvePnPRansac(
                                obj_abs, img_pts, self.K, self.dist,
                                flags=cv2.SOLVEPNP_ITERATIVE,
                                iterationsCount=80
                            )
                            if not ok:
                                continue

                            R,_ = cv2.Rodrigues(rvec_g)
                            C_world = (-R.T @ tvec_g).flatten()
                            # top-down 보정

                            cam_positions.append(C_world)
                            cam_rotations.append(R)

                        # =====================================================
                        # Multi-marker pose fusion
                        # =====================================================
                        if len(cam_positions) > 0:

                            cam_positions = np.array(cam_positions)
                            mean_pos = np.mean(cam_positions, axis=0)

                            R_stack = np.stack(cam_rotations)
                            R_mean = R_stack.mean(axis=0)
                            U,_,Vt = np.linalg.svd(R_mean)
                            R_mean = U @ Vt

                            ang = np.degrees(self._rotation_matrix_to_euler(R_mean))
                            roll, pitch, yaw = np.degrees(self._rotation_matrix_to_euler(R_mean))
                            yaw = -yaw
                            ang = (roll, pitch, yaw)

                            if not hasattr(self,'filtered_pos') or self.filtered_pos is None:
                                self.filtered_pos = mean_pos
                            else:
                                self.filtered_pos = 0.85*self.filtered_pos + 0.15*mean_pos

                            stable = self.filtered_pos
                            xy_angle = (stable[0], stable[1], ang[2])
                            self.current_position = xy_angle

                            # Camera pose print
                            cv2.putText(color,
                                        f"Cam(World): X={stable[0]:.2f}, Y={stable[1]:.2f}, Z={stable[2]:.2f}",
                                        (10, 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0,255,255), 2)

                            cv2.putText(color,
                                        f"Yaw={ang[2]:.1f} deg",
                                        (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0,255,255), 2)

                            # UDP publish
                            msg = f"{stable[0]},{stable[1]},{stable[2]},{ang[0]},{ang[1]},{ang[2]}"
                            self.sock.sendto(msg.encode(), self.server_addr)

                            if self.publish_point:
                                self.publish_point(xy_angle,"current")

                            self.last_R_world = R_mean
                            self.last_t_world = stable

                            birdseye = self._create_birdseye_view(stable[0], stable[1], ang[2])
                            if self.show_bev:
                                cv2.imshow("Bird's Eye View", birdseye)

                # Target detection → world coordinates + Print
                if hasattr(self, 'last_R_world') and hasattr(self, 'last_cam_world'):

                    det_results = self.det_model(color, verbose=False)

                    for r in det_results:
                        for box in r.boxes:

                            cls_id = int(box.cls[0])
                            cls_name = self.det_model.names.get(cls_id)

                            if cls_name not in self.TARGET_CLASSES:
                                continue

                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            cx = (x1+x2)//2
                            cy = (y1+y2)//2

                            d_m = self._depth(depth_img, cx, cy)
                            if d_m is None or d_m > self.MAX_DEPTH_M:
                                continue

                            fx,fy = self.K[0,0], self.K[1,1]
                            cx0,cy0 = self.K[0,2], self.K[1,2]

                            Xc = (cx-cx0)*d_m/fx
                            Yc = (cy-cy0)*d_m/fy
                            Zc = d_m
                            Pc = np.array([Xc,Yc,Zc])

                            # yaw offset (카메라 yaw 보정이 정말 필요하면 유지)
                            da = math.radians(self.offset)
                            Rz = np.array([
                                [math.cos(da), -math.sin(da), 0],
                                [math.sin(da),  math.cos(da), 0],
                                [0,0,1]
                            ])
                            Pc_rot = Rz @ Pc

                            # ---- 올바른 world 변환: Xw = C + Rᵀ * Pc ----
                            Pw = self.last_cam_world + self.last_R_world.T @ Pc_rot

                            # Target world 좌표 화면 출력
                            cv2.putText(color,
                                        f"Target({cls_name}): ({Pw[0]:.2f},{Pw[1]:.2f},{Pw[2]:.2f})",
                                        (x1, max(0, y1-20)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55, (255,200,0), 2)

                            # UDP publish
                            msg = f"{cls_name},{Pw[0]},{Pw[1]},{Pw[2]}"
                            self.target_sock.sendto(msg.encode(), self.target_addr)

                            if self.publish_point:
                                self.publish_point(Pw,"target")


                # Display
                if self.show_windows:
                    cv2.imshow("Camera Stream", color)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.pipeline.stop()
            self.sock.close()
            self.target_sock.close()
            if self.show_windows:
                cv2.destroyAllWindows()


# Standalone run
if __name__ == "__main__":
    rospy.init_node("rs_cam")
    streamer = RealSenseLocalizationStreamer(show_windows=True)
    streamer.run()
