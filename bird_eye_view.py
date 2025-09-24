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
center_x = (x_max + x_min)/2.0
center_y = (y_max + y_min)/2.0


origin_shift = _raw_marker_positions[0] - origin_offset
MARKER_WORLD_COORDS = (_raw_marker_positions)
MARKER_LENGTH_M = 0.20

MAP_PARAMS = {
    'size_px': 500,
    'scale': 200, # 맵 스케일 증가
}
map_w, map_h = MAP_PARAMS['size_px'], MAP_PARAMS['size_px']
origin_world = np.array([x_min, y_min])
origin_x_px = int((map_w / 2) - (center_x - x_min) * MAP_PARAMS['scale'])
origin_y_px = int((map_h / 2) + (center_y - y_min) * MAP_PARAMS['scale'])
MAP_PARAMS['origin_px'] = (origin_x_px, origin_y_px)

# --- 헬퍼 함수 (camera_stream_test.py에서 가져옴) ---
def world_to_map_coords(world_xy):
    wx, wy = world_xy
    ox, oy = MAP_PARAMS['origin_px']
    scale = MAP_PARAMS['scale']
    map_x = int(ox + (wx - origin_world[0]) * scale)
    map_y = int(oy - (wy - origin_world[1]) * scale)
    return map_x, map_y

def draw_static_map_elements(canvas):
    # --- MODIFICATION 2: 원점 재정의 및 X, Y축 그리기 ---
    marker_gap_m = 0.30 # 축을 그리기 위한 간격 값

    # 1. 새로운 월드 원점 (0,0)을 마커 0의 오른쪽 아래로 정의
    # 마커 0의 월드 좌표: _raw_marker_positions[0] -> (0.9, 0.0)
    # 새로운 원점의 월드 좌표: (0.9 + 0.3, 0.0 - 0.3) -> (1.2, -0.3)
    world_origin_coord = _raw_marker_positions[0][:2] + np.array([marker_gap_m, -marker_gap_m])
    
    # 2. 월드 원점을 맵(픽셀) 좌표로 변환
    origin_px = world_to_map_coords(world_origin_coord)
    
    # 3. 맵의 경계에 맞춰 축의 시작점과 끝점 계산 (월드 좌표 기준)
    xs = [pos[0] for pos in _raw_marker_positions.values()]
    ys = [pos[1] for pos in _raw_marker_positions.values()]
    x_axis_start_w = (min(xs) - marker_gap_m, world_origin_coord[1])
    x_axis_end_w = (max(xs) + marker_gap_m, world_origin_coord[1])
    y_axis_start_w = (world_origin_coord[0], min(ys) - marker_gap_m)
    y_axis_end_w = (world_origin_coord[0], max(ys) + marker_gap_m)

    # 4. 축 좌표들을 픽셀 좌표로 변환
    x_axis_start_px = world_to_map_coords(x_axis_start_w)
    x_axis_end_px = world_to_map_coords(x_axis_end_w)
    y_axis_start_px = world_to_map_coords(y_axis_start_w)
    y_axis_end_px = world_to_map_coords(y_axis_end_w)

    # 5. 축 그리기 (화살표 포함)
    cv2.arrowedLine(canvas, y_axis_start_px, y_axis_end_px, (0, 150, 0), 2) # Y축 (초록색)
    cv2.arrowedLine(canvas, x_axis_start_px, x_axis_end_px, (0, 0, 150), 2) # X축 (빨간색)
    cv2.putText(canvas, "Y", (y_axis_end_px[0] + 10, y_axis_end_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
    cv2.putText(canvas, "X", (x_axis_end_px[0], x_axis_end_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)

    # 6. 원점(0,0) 표시
    cv2.circle(canvas, origin_px, 5, (0, 0, 0), -1)
    cv2.putText(canvas, "(0,0)", (origin_px[0] + 5, origin_px[1] + 15), 
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

    # # BEV 창 먼저 띄우기
    # cv2.imshow("BEV Data Receiver", map_canvas)
    # cv2.waitKey(1)

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
            
            # 카메라 좌표를 점 옆에 바로 표시
            coord_text = f"({pos_x:.3f},{pos_y:.3f})"
            cv2.putText(bev_image, coord_text, (cam_px + 10, cam_px-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            cv2.imshow("Bird Eye View map", bev_image)

        except (ValueError, IndexError):
            print(f"Received malformed data: {data_string}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()