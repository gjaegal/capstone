#!/usr/bin/env python3
import cv2
import numpy as np
import socket
import math

# ------------------------------------------------------------
# 1) 새 환경: 바닥 마커 기준 월드 좌표 (3×4 grid)
# ------------------------------------------------------------
MARKER_WORLD_COORDS = {
    0:  np.array([0.0, 0.0, 0.0]),
    1:  np.array([2.0, 0.0, 0.0]),
    2:  np.array([4.0, 0.0, 0.0]),
    3:  np.array([6.0, 0.0, 0.0]),

    4:  np.array([0.0, 1.5, 0.0]),
    5:  np.array([2.0, 1.5, 0.0]),
    6:  np.array([4.0, 1.5, 0.0]),
    7:  np.array([6.0, 1.5, 0.0]),

    8:  np.array([0.0, 3.0, 0.0]),
    9:  np.array([2.0, 3.0, 0.0]),
    10: np.array([4.0, 3.0, 0.0]),
    11: np.array([6.0, 3.0, 0.0]),
}

# ------------------------------------------------------------
# 2) BEV 설정
# ------------------------------------------------------------
MAP_SIZE = 900                # px
SCALE = 100                   # 1m = 100px
ORIGIN_PX = MAP_SIZE // 2     # 화면 정중앙 px

# ----- BEV 중앙 = 마커 5와 6의 중점 -----
center_56 = (MARKER_WORLD_COORDS[5][:2] + MARKER_WORLD_COORDS[6][:2]) / 2.0
# center_56 = (3.0, 1.5)

# ------------------------------------------------------------
# 3) 월드 → BEV 변환 함수
# ------------------------------------------------------------
def world_to_map(world_xy):
    wx, wy = world_xy

    # BEV 중앙을 world 좌표 (3,1.5)에 맞춤
    wx_shift = wx - center_56[0]
    wy_shift = wy - center_56[1]

    # 픽셀 변환
    x = int(ORIGIN_PX + wx_shift * SCALE)
    y = int(ORIGIN_PX - wy_shift * SCALE)
    return x, y

# ------------------------------------------------------------
# 4) 기본 맵 및 grid 그리기
# ------------------------------------------------------------
def draw_static(canvas):
    H, W = canvas.shape[:2]

    X_MAX = 6.0
    Y_MAX = 3.0
    STEP = 0.5

    # grid lines
    x_tick = 0.0
    while x_tick <= X_MAX + 1e-9:
        (x1, y1) = world_to_map((x_tick, 0.0))
        (x2, y2) = world_to_map((x_tick, Y_MAX))
        cv2.line(canvas, (x1, y1), (x2, y2),
                 (200, 200, 200), 1)
        x_tick += STEP

    y_tick = 0.0
    while y_tick <= Y_MAX + 1e-9:
        (x1, y1) = world_to_map((0.0, y_tick))
        (x2, y2) = world_to_map((X_MAX, y_tick))
        cv2.line(canvas, (x1, y1), (x2, y2),
                 (200, 200, 200), 1)
        y_tick += STEP

    # draw markers
    marker_half_px = int(0.20 * SCALE / 2)
    for mid, w in MARKER_WORLD_COORDS.items():
        mx, my = world_to_map((w[0], w[1]))
        cv2.rectangle(canvas,
                      (mx - marker_half_px, my - marker_half_px),
                      (mx + marker_half_px, my + marker_half_px),
                      (80, 80, 80), -1)
        cv2.putText(canvas, str(mid),
                    (mx + 6, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 50, 50), 2)
        
    special_ids = [0, 3, 8, 11]
    for sid in special_ids:
        wx, wy, _ = MARKER_WORLD_COORDS[sid]
        px, py = world_to_map((wx, wy))
        cv2.putText(canvas,
                    f"({wx:.1f},{wy:.1f})",
                    (px + 12, py + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (50, 50, 50), 2
                    )

# ------------------------------------------------------------
# 5) 메인 루프
# ------------------------------------------------------------
def main():
    # sockets
    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cam_sock.bind(('localhost', 12345))
    cam_sock.settimeout(0.02)

    tgt_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tgt_sock.bind(('localhost', 12346))
    tgt_sock.settimeout(0.02)

    path_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    path_sock.bind(('localhost', 12348))
    path_sock.settimeout(0.02)

    # BEV base
    base = np.full((MAP_SIZE, MAP_SIZE, 3),
                   (240,240,240), dtype=np.uint8)
    draw_static(base)

    last_cam = None
    last_target = None
    path_points = []

    while True:
        # ---------------- camera pose ----------------
        try:
            s, _ = cam_sock.recvfrom(1024)
            x, y, z, yaw_deg = map(float, s.decode().split(','))
            last_cam = (x, y, z, yaw_deg)
        except socket.timeout:
            pass

        # ---------------- target pose ----------------
        try:
            s, _ = tgt_sock.recvfrom(1024)
            parts = s.decode().split(',')
            cls = parts[0]
            tx,ty,tz = map(float, parts[1:])
            last_target = (cls, tx, ty, tz)
        except socket.timeout:
            pass

        # ---------------- path ----------------
        try:
            p, _ = path_sock.recvfrom(4096)
            coords = p.decode().split(';')
            path_points = [tuple(map(float,c.split(',')))
                           for c in coords if c]
        except socket.timeout:
            pass

        # ---------------- draw BEV ----------------
        bev = base.copy()

        # robot
        if last_cam is not None:
            x,y,z,yaw = last_cam
            px,py = world_to_map((x,y))

            cv2.circle(bev,(px,py),8,(0,0,255),-1)
            cv2.putText(bev, f"Me ({x:.1f},{y:.1f})",
                        (px+10, py-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,0),2)

            theta = math.radians(yaw) - math.pi/2
            ex = x + 0.5*math.cos(theta)
            ey = y + 0.5*math.sin(theta)
            ex_px, ey_px = world_to_map((ex,ey))
            cv2.line(bev,(px,py),(ex_px,ey_px),(0,0,255),2)

        # target
        if last_target is not None:
            cls, tx,ty,tz = last_target
            px,py = world_to_map((tx,ty))
            cv2.circle(bev,(px,py),8,(255,0,0),-1)
            cv2.putText(bev, f"{cls} ({tx:.1f},{ty:.1f})",
                        (px+10,py-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,
                        (255,0,0),2)

        # path
        if path_points:
            pts = np.array([world_to_map((x,y))
                            for (x,y) in path_points], np.int32)
            cv2.polylines(bev, [pts], False, (0,255,255), 2)

        # ---------------- show ----------------
        cv2.imshow("Bird's Eye View", bev)
        cv2.moveWindow("Bird's Eye View", 1200,200)

        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cam_sock.close()
    tgt_sock.close()
    path_sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
