# bird_eye_view_map.py

import cv2
import numpy as np
import socket
import math

# --- 설정 (camera_stream.py와 동일하게 유지) ---
# --- camera_stream_test.py와 동일하게: 3행 x 4열, 0번은 오른쪽 아래 ---
marker_gap_m = 0.30
origin_offset = np.array([marker_gap_m, marker_gap_m, 0.0], dtype=np.float32)

_raw_marker_positions = {
    0:  np.array([0.9, 0.0, 0.0]),
    1:  np.array([0.6, 0.0, 0.0]),
    2:  np.array([0.3, 0.0, 0.0]),
    3:  np.array([0.0, 0.0, 0.0]),

    4:  np.array([0.9, 0.3, 0.0]),
    5:  np.array([0.6, 0.3, 0.0]),
    6:  np.array([0.3, 0.3, 0.0]),
    7:  np.array([0.0, 0.3, 0.0]),

    8:  np.array([0.9, 0.6, 0.0]),
    9:  np.array([0.6, 0.6, 0.0]),
    10: np.array([0.3, 0.6, 0.0]),
    11: np.array([0.0, 0.6, 0.0]),
}

# 0번 마커 중심 - origin_offset을 원점으로 맞춤
xs = [pos[0] for pos in _raw_marker_positions.values()]
ys = [pos[1] for pos in _raw_marker_positions.values()]
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)

origin_shift = _raw_marker_positions[0] - origin_offset
MARKER_WORLD_COORDS = _raw_marker_positions
MARKER_LENGTH_M = 0.20

MAP_PARAMS = {
    'size_px': 500,
    'scale': 150,
}
map_w, map_h = MAP_PARAMS['size_px'], MAP_PARAMS['size_px']
origin_world = np.array([x_min, y_min])
MAP_PARAMS['origin_px'] = (50, map_h - 50)


# --- 헬퍼 함수 (camera_stream_test.py에서 가져옴) ---
def world_to_map_coords(world_xy):
    wx, wy = world_xy
    ox, oy = MAP_PARAMS['origin_px']
    scale = MAP_PARAMS['scale']
    map_x = int(ox + (wx - origin_world[0]) * scale)
    map_y = int(oy - (wy - origin_world[1]) * scale)
    return map_x, map_y

def draw_static_map_elements(canvas):
    origin_px = MAP_PARAMS['origin_px']
    cv2.circle(canvas, origin_px, 5, (0, 0, 0), -1)
    cv2.putText(canvas, "(0,0)", (origin_px[0] + 5, origin_px[1] + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    for mid, world_coord in MARKER_WORLD_COORDS.items():
        marker_px, marker_py = world_to_map_coords((world_coord[0], world_coord[1]))
        marker_size_px = int(MARKER_LENGTH_M * MAP_PARAMS['scale'] / 2)
        cv2.rectangle(canvas, (marker_px - marker_size_px, marker_py - marker_size_px),
                      (marker_px + marker_size_px, marker_py + marker_size_px), (0, 0, 0), -1)
        cv2.putText(canvas, str(mid), (marker_px + 5, marker_py - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

# --- 메인 실행 부분 ---
def main():
    # UDP 소켓 생성 및 바인딩
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 12345)
    sock.bind(server_address)

    # 고정된 요소가 그려진 맵 캔버스 생성
    map_canvas = np.full((MAP_PARAMS['size_px'], MAP_PARAMS['size_px'], 3), 
                        (240, 240, 240), dtype=np.uint8)
    draw_static_map_elements(map_canvas)

    print("Listening for camera position data on localhost:12345...")

    while True:
        # 데이터 수신 대기 (데이터가 올 때까지 여기서 멈춤)
        data, _ = sock.recvfrom(1024)
        data_string = data.decode('utf-8')

        try:
            # 수신한 문자열 파싱: "x,y,z,roll,pitch,yaw"
            parts = [float(p) for p in data_string.split(',')]
            pos_x, pos_y, _, _, _, yaw_deg = parts

            # 맵 업데이트
            bev_image = map_canvas.copy()
            cam_px, cam_py = world_to_map_coords((pos_x, pos_y))
            
            # 카메라 방향(yaw) 표시
            yaw_rad = math.radians(yaw_deg)
            dir_len = 0.2 * MAP_PARAMS['scale']
            dir_end_x = int(cam_px + dir_len * math.cos(yaw_rad))
            dir_end_y = int(cam_py - dir_len * math.sin(yaw_rad))
            
            # 맵에 카메라 위치와 방향 그리기
            cv2.circle(bev_image, (cam_px, cam_py), 7, (0, 0, 255), -1)
            cv2.line(bev_image, (cam_px, cam_py), (dir_end_x, dir_end_y), (0, 0, 255), 2)

            cv2.imshow("Bird's-Eye View (Data Receiver)", bev_image)

        except (ValueError, IndexError):
            print(f"Received malformed data: {data_string}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()