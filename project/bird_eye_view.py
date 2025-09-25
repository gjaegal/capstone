# bird_eye_view.py

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
    'size_px': 900,     # 출력 창 크기
    'scale': 100,       # 1m = 100px
}
MAP_PARAMS['origin_px'] = (MAP_PARAMS['size_px']//2, MAP_PARAMS['size_px']//2)  # 화면 정중앙

# --- 시야 중심: ID5 & ID6의 중점 (화면 중앙이 여기를 보게 함) ---
_center_56 = (MARKER_WORLD_COORDS[5][:2] + MARKER_WORLD_COORDS[6][:2]) / 2.0  # (2.4, 1.5)

# --- grid 원점: ID0 (라벨/눈금 0m 기준) ---
_origin_id0 = MARKER_WORLD_COORDS[0][:2]  # (0.0, 0.0)

# --- 월드→스크린 변환 (화면 중심 = 중점(5,6), X좌우반전 포함) ---
def world_to_map_coords(world_xy):
    wx, wy = world_xy
    # 화면 중심 기준으로 시프트 + X축 좌우 반전
    wx_shift = -(wx - _center_56[0])
    wy_shift =  (wy - _center_56[1])

    ox, oy = MAP_PARAMS['origin_px']
    s = MAP_PARAMS['scale']
    x = int(ox + wx_shift * s)
    y = int(oy - wy_shift * s)  # y는 위로 증가
    return x, y

# --- 축/눈금(axes-only), 마커들 ---
def draw_static_map_elements(canvas):
    H, W = canvas.shape[:2]
    cx, cy = MAP_PARAMS['origin_px']

    # ----- Grid: ID0을 원점으로 한 축 눈금 (1m 간격) -----
    X_MAX = 6  # x축 0~6m
    Y_MAX = 3  # y축 0~3m

    # 원점(=ID0) 픽셀 좌표
    id0_px, id0_py = world_to_map_coords((_origin_id0[0], _origin_id0[1]))

    # 축 라인
    cv2.line(canvas, (0, id0_py), (W, id0_py), (0, 0, 0), 1)   # x축
    cv2.line(canvas, (id0_px, 0), (id0_px, H), (0, 0, 0), 1)   # y축

    # x축 눈금: 0~X_MAX m (ID0 기준, 좌우반전이므로 값이 커질수록 왼쪽으로 감)
    for i in range(0, X_MAX+1):
        tx, ty = world_to_map_coords((_origin_id0[0] + i, _origin_id0[1]))
        cv2.line(canvas, (tx, id0_py - 5), (tx, id0_py + 5), (0, 0, 0), 1)
        cv2.putText(canvas, f"{i}m", (tx - 12, id0_py + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # y축 눈금: 0~Y_MAX m (ID0 기준, 위로 증가)
    for j in range(0, Y_MAX+1):
        tx, ty = world_to_map_coords((_origin_id0[0], _origin_id0[1] + j))
        cv2.line(canvas, (id0_px - 5, ty), (id0_px + 5, ty), (0, 0, 0), 1)
        cv2.putText(canvas, f"{j}m", (id0_px + 10, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 원점(ID0) 표시
    cv2.circle(canvas, (id0_px, id0_py), 6, (0, 0, 255), -1)
    cv2.putText(canvas, "ID0 (0,0)", (id0_px + 10, id0_py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # ----- 마커들 (좌우반전된 배치로 보임 / 텍스트는 정상) -----
    marker_half_px = int(MARKER_LENGTH_M * MAP_PARAMS['scale'] / 2)
    for mid, w in MARKER_WORLD_COORDS.items():
        mx, my = world_to_map_coords((w[0], w[1]))
        cv2.rectangle(canvas, (mx - marker_half_px, my - marker_half_px),
                      (mx + marker_half_px, my + marker_half_px), (50, 50, 50), -1)
        cv2.putText(canvas, str(mid), (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 50), 3)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 12345)
    sock.bind(server_address)
    sock.settimeout(0.02)  # ✅ 타임아웃으로 non-blocking: q 입력 즉시 반응

    base = np.full((MAP_PARAMS['size_px'], MAP_PARAMS['size_px'], 3), (240, 240, 240), dtype=np.uint8)
    draw_static_map_elements(base)

    print("Listening for camera position data on localhost:12345... (press 'q' to quit)")

    bev = base.copy()  # 마지막 프레임(데이터 없어도 화면 띄움)
    while True:
        try:
            s, _ = sock.recvfrom(1024)
            s = s.decode('utf-8').strip()
            # "x,y,z,roll,pitch,yaw"
            x, y, z, roll, pitch, yaw_deg = [float(p) for p in s.split(',')]

            bev = base.copy()

            # 시작점
            sx, sy = world_to_map_coords((x, y))

            # 카메라 좌표 m단위로 표시
            cv2.putText(bev, f"({x:.2f}, {y:.2f})", (sx + 10, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

            # 월드에서 끝점 계산 후 변환(반전/시야중심을 올바르게 반영)
            arrow_len_m = 0.5  # 0.5m 길이 화살표
            theta = math.radians(yaw_deg) + (-math.pi/2)
            ex_w = x + arrow_len_m * math.cos(theta)
            ey_w = y + arrow_len_m * math.sin(theta)
            ex, ey = world_to_map_coords((ex_w, ey_w))

            cv2.circle(bev, (sx, sy), 7, (0, 0, 255), -1)
            cv2.line(bev, (sx, sy), (ex, ey), (0, 0, 255), 2)

        except socket.timeout:
            # 데이터가 없어도 베이스를 계속 보여줌
            pass
        except Exception as e:
            print(f"Bad packet | err={e}")

        cv2.imshow("Bird's-Eye View", bev)

        # ✅ 'q' 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 창을 직접 닫았을 때도 종료
        if cv2.getWindowProperty("Bird's-Eye View", cv2.WND_PROP_VISIBLE) < 1:
            break

    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
