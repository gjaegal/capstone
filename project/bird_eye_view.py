#!/usr/bin/env python3
# bird_eye_view.py (로봇/타겟/장애물 BEV 통합 표시)

import cv2
import numpy as np
import socket
import math

# --- 실제 맵 좌표 (단위: m) ---
MARKER_WORLD_COORDS = {
    0:  np.array([0.0, 0.0, 0.0]),
    1:  np.array([1.5, 0.0, 0.0]),
    2:  np.array([3.2, 0.0, 0.0]),
    3:  np.array([5.0, 0.0, 0.0]),
    4:  np.array([0.0, 1.5, 0.0]),
    5:  np.array([1.6, 1.5, 0.0]),
    6:  np.array([3.2, 1.5, 0.0]),
    7:  np.array([4.8, 1.5, 0.0]),
    8:  np.array([0.0, 3.0, 0.0]),
    9:  np.array([1.5, 3.0, 0.0]),
    10: np.array([3.1, 3.0, 0.0]),
    11: np.array([5.0, 3.0, 0.0]),
}
MARKER_LENGTH_M = 0.20

# --- 맵/렌더링 설정 ---
MAP_PARAMS = {
    'size_px': 900,
    'scale': 100,  # 1m = 100px
}
MAP_PARAMS['origin_px'] = (MAP_PARAMS['size_px']//2, MAP_PARAMS['size_px']//2)

# --- 시야 중심: ID5 & ID6의 중점 (화면 중앙 기준) ---
_center_56 = (MARKER_WORLD_COORDS[5][:2] + MARKER_WORLD_COORDS[6][:2]) / 2.0
_origin_id0 = MARKER_WORLD_COORDS[0][:2]

# --- 월드→맵 변환 ---
def world_to_map_coords(world_xy):
    wx, wy = world_xy
    wx_shift = -(wx - _center_56[0])
    wy_shift =  (wy - _center_56[1])
    ox, oy = MAP_PARAMS['origin_px']
    s = MAP_PARAMS['scale']
    x = int(ox + wx_shift * s)
    y = int(oy - wy_shift * s)
    return x, y

# --- 기본 맵 축/눈금/마커 ---
def draw_static_map_elements(canvas):
    H, W = canvas.shape[:2]

    # ----- Grid 범위(월드 단위, m) -----
    X_MAX = 6.0   # x축 0~6 m
    Y_MAX = 3.0   # y축 0~3 m
    STEP  = 0.5   # 0.5 m 간격

    # 원점(=ID0) 픽셀 좌표
    id0_px, id0_py = world_to_map_coords((MARKER_WORLD_COORDS[0][0], MARKER_WORLD_COORDS[0][1]))

    # ----- 0.5 m 그리드 (연한 회색), 1 m 라인(짙은 회색) -----
    # 세로선: x = 0, 0.5, 1.0, ..., X_MAX
    x_tick = 0.0
    while x_tick <= X_MAX + 1e-9:
        (x1, y1) = world_to_map_coords((x_tick, 0.0))
        (x2, y2) = world_to_map_coords((x_tick, Y_MAX))
        color = (210, 210, 210)
        thickness = 1 if (abs(x_tick - round(x_tick)) > 1e-9) else 1
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
        x_tick += STEP

    # 가로선: y = 0, 0.5, 1.0, ..., Y_MAX
    y_tick = 0.0
    while y_tick <= Y_MAX + 1e-9:
        (x1, y1) = world_to_map_coords((0.0,  y_tick))
        (x2, y2) = world_to_map_coords((X_MAX, y_tick))
        color = (150, 150, 150)
        thickness = 1 if (abs(y_tick - round(y_tick)) > 1e-9) else 1
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
        y_tick += STEP

    # ----- 축/눈금(1 m 라벨만) -----
    # x축(원점 y 위치) / y축(원점 x 위치)
    cv2.line(canvas, (0, id0_py), (W, id0_py), (120, 120, 120), 1)
    cv2.line(canvas, (id0_px, 0), (id0_px, H), (120, 120, 120), 1)

    # 1 m 라벨
    for i in range(0, int(X_MAX) + 1):
        tx, ty = world_to_map_coords((float(i), 0.0))
        cv2.putText(canvas, f"{i}m", (tx - 12, id0_py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    for j in range(0, int(Y_MAX) + 1):
        tx, ty = world_to_map_coords((0.0, float(j)))
        cv2.putText(canvas, f"{j}m", (id0_px + 10, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # ----- 원점(ID0) 마커 -----
    cv2.circle(canvas, (id0_px, id0_py), 6, (0, 0, 255), -1)
    cv2.putText(canvas, "ID0 (0,0)", (id0_px + 10, id0_py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # ----- 아루코 마커 사각형/라벨 -----
    marker_half_px = int(MARKER_LENGTH_M * MAP_PARAMS['scale'] / 2)
    for mid, w in MARKER_WORLD_COORDS.items():
        mx, my = world_to_map_coords((w[0], w[1]))
        cv2.rectangle(canvas, (mx - marker_half_px, my - marker_half_px),
                      (mx + marker_half_px, my + marker_half_px), (60, 60, 60), -1)
        cv2.putText(canvas, str(mid), (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 50), 3)


# --- 메인 ---
def main():
    # UDP 소켓 초기화
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('localhost', 12345))
    sock.settimeout(0.02)

    target_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_sock.bind(('localhost', 12346))
    target_sock.settimeout(0.02)

    obstacle_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    obstacle_sock.bind(('localhost', 12347))
    obstacle_sock.settimeout(0.02)


    # >>> 추가 부분 (A* 경로 수신용) <<<
    path_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    path_sock.bind(('localhost', 12348))   # controller4.py에서 보낼 포트
    path_sock.settimeout(0.02)
    path_points = []

    base = np.full((MAP_PARAMS['size_px'], MAP_PARAMS['size_px'], 3),
                   (240, 240, 240), dtype=np.uint8)
    draw_static_map_elements(base)

    print("Listening for camera (12345), target (12346), obstacles (12347)... Press 'q' to quit.")
    bev = base.copy()
    last_cam_pose = None
    last_target = None
    obstacles = []

    while True:
        try:
            ######################## 카메라 방향 표시 ######################
            s, _ = sock.recvfrom(1024)
            x, y, z, roll, pitch, yaw_deg = [float(p) for p in s.decode('utf-8').split(',')]
            last_cam_pose = (x, y, z, roll, pitch, yaw_deg)
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Bad packet (camera): {e}")

            ############################### 타겟 표시 ##############################
        try:
            t, _ = target_sock.recvfrom(1024)
            parts = t.decode('utf-8').split(',')
            if len(parts) == 4:
                cls_name = parts[0]
                tx, ty, tz = map(float, parts[1:])
            else:
                cls_name = "object"
                tx, ty, tz = map(float, parts[:3])
            last_target = (tx, ty, tz) # 최신 타겟 좌표 저장
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Bad packet (target): {e}")

        bev = base.copy()

        if last_cam_pose is not None:
            x, y, z, roll, pitch, yaw_deg = last_cam_pose
            sx, sy = world_to_map_coords((x, y))
            cv2.putText(bev, f"Me ({x:.1f}, {y:.1f})", (sx+10, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            arrow_len_m = 0.5
            theta = math.radians(yaw_deg) + (-math.pi/2)
            ex_w = x + arrow_len_m * math.cos(theta)
            ey_w = y + arrow_len_m * math.sin(theta)
            ex, ey = world_to_map_coords((ex_w, ey_w))

            cv2.circle(bev, (sx, sy), 7, (0,0,255), -1)
            cv2.line(bev, (sx, sy), (ex, ey), (0,0,255), 2)

        if last_target is not None:
            tx, ty, tz = last_target
            px, py = world_to_map_coords((tx, ty))
            cv2.circle(bev, (px, py), 8, (255, 0, 0), -1)
            cv2.putText(bev, f"{cls_name} ({tx:.1f}, {ty:.1f})", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

################################ 장애물 표시 ################################
        try:
            o, _ = obstacle_sock.recvfrom(1024)
            ox, oy, oz = [float(p) for p in o.decode('utf-8').split(',')]
            px, py = world_to_map_coords((ox, oy))
            obstacles.append((px, py))
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Bad packet (obstacle): {e}")

        # 누적된 장애물 그리기 (최근 200개만 유지)
        for (px, py) in obstacles[-200:]:
            cv2.circle(bev, (px, py), 4, (0, 0, 0), -1)


################################ 경로 표시 ################################
        try:
            p, _ = path_sock.recvfrom(4096)
            coords = p.decode('utf-8').split(';')   # "x1,y1;x2,y2;..." 형태
            path_points = [tuple(map(float, c.split(','))) for c in coords if c]
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Bad packet (path): {e}")

        # A* 경로 시각화
        if path_points:
            pts = np.array([world_to_map_coords((x, y)) for (x, y) in path_points], np.int32)
            if len(pts) > 1:
                cv2.polylines(bev, [pts], False, (0, 255, 255), 2)  # 노란색 선으로 연결
            else:
                for (x, y) in pts:
                    px, py = world_to_map_coords((x, y))
                    cv2.circle(bev, (px, py), 4, (0, 255, 255), -1)


        cv2.imshow("Bird's-Eye View", bev)
        cv2.moveWindow("Bird's-Eye View", 1200, 200) # 팝업 창 위치 설정
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Bird's-Eye View", cv2.WND_PROP_VISIBLE) < 1:
            break




    sock.close()
    target_sock.close()
    obstacle_sock.close()
    path_sock.close() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
