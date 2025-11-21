#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import tf
from astar import astar, create_grid, discretize
import socket
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient

# ---- 간단한 2D 벡터/회전 유틸 ----
def rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s), (s, c))

def mat2_mul_vec(M, v):
    return (M[0][0]*v[0] + M[0][1]*v[1],
            M[1][0]*v[0] + M[1][1]*v[1])

def vec_add(a, b):
    return (a[0]+b[0], a[1]+b[1])

def vec_sub(a, b):
    return (a[0]-b[0], a[1]-b[1])


class ServingRobotController:
    def __init__(self):
        rospy.init_node('serving_robot_controller')
        
        # 퍼블리셔/구독자
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
        self.current_point_sub = rospy.Subscriber('/current_point', Point, self.current_point_callback)
        self.target_point_sub  = rospy.Subscriber('/target_point',  Point, self.target_point_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)  # 오도메트리 구독(누적보정 핵심)

        # BEV 경로 전송용 UDP
        self.path_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.path_addr = ('localhost', 12348)

        # 상태/좌표
        self.state = "SEARCH"
        self.target_found = False
        self.current_position = [0.0, 0.0]
        self.current_yaw = 0.0
        self.target_position = [0.5, 0.5]


        self.target_locked = False
        self.locked_target_position = None

        # 오도메트리 누적용
        self.last_odom_pos = None
        self.last_odom_yaw = None

        # (옵션) 정렬 파라미터(현재는 사용 안 함: has_align=False)
        self.has_align = False
        self.R_align = ((1.0, 0.0), (0.0, 1.0))
        self.t_align = (0.0, 0.0)

        # 탐색 회전 방향 토글
        self.search_initialized = False
        self.search_left = True

        self.twist = Twist()
        rospy.loginfo("서빙 로봇 컨트롤러 시작 - YOLO+RealSense 모드")


    # ----------------------------- 콜백 -----------------------------

    def target_point_callback(self, msg):

        if self.target_locked:
            return

        rospy.loginfo(f"타겟 좌표: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
        self.locked_target_position = [msg.x, msg.y]  
        self.target_position = self.locked_target_position  
        self.target_found = True
        self.target_locked = True   

    def current_point_callback(self, msg):
        if self.current_position != [msg.x, msg.y]:
            rospy.loginfo(f"현재 좌표(vision): x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
            self.current_position = [msg.x, msg.y]

    def odom_callback(self, msg):
        # 오도메트리 기반 누적 보정
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        odom_now = (p.x, p.y)

        if self.last_odom_pos is None:
            self.last_odom_pos = odom_now
            self.last_odom_yaw = yaw
            return

        # Δ (odom)
        # delta_odom = vec_sub(odom_now, self.last_odom_pos)
        # delta_yaw  = self._normalize_angle(yaw - self.last_odom_yaw)

        # if not self.has_align:
        #     # 정렬(vision)이 없을 때: 현재 추정 yaw 기준 회전행렬로 Δ를 world에 누적
        #     R = rot2d(self.current_yaw)
        #     d_world = mat2_mul_vec(R, delta_odom)
        #     self.current_position = vec_add(self.current_position, d_world)
        #     self.current_yaw = self._normalize_angle(self.current_yaw + delta_yaw)
        # else:
        #     # 정렬이 있다면: 미리 구한 R_align로 Δ를 world에 사상(옵션)
        #     d_world = mat2_mul_vec(self.R_align, delta_odom)
        #     self.current_position = vec_add(self.current_position, d_world)
        #     self.current_yaw = self._normalize_angle(self.current_yaw + delta_yaw)

        # self.last_odom_pos = odom_now
        # self.last_odom_yaw = yaw


    # ----------------------------- 유틸 -----------------------------

    def get_current_yaw(self):
        try:
            odom_msg = rospy.wait_for_message("/odom", Odometry, timeout=1.0)
            q = odom_msg.pose.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return yaw
        except Exception as e:
            rospy.logwarn(f"[SEARCH] Failed to get yaw: {e}")
            return 0.0

    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, target, current):
        return self._normalize_angle(target - current)

    def yaw_to_dir(self, yaw_deg):
        yaw_norm = yaw_deg % 360.0
        if   45 <= yaw_norm < 135:   return (0, 1)    # 동
        elif 135 <= yaw_norm < 225:  return (1, 0)    # 남
        elif 225 <= yaw_norm < 315:  return (0, -1)   # 서
        else:                        return (-1, 0)   # 북

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)


    # ----------------------------- 탐색 모드 -----------------------------

    def search_mode(self):
        # --- 타겟 이미 탐지된 경우 ---
        if self.target_found:
            rospy.loginfo("[SEARCH] 타겟 감지됨 → APPROACH 모드 대기 중...")

        base_angle_deg = 90.0
        angular_speed = 0.5

        if not self.search_initialized:
            angle_deg = base_angle_deg / 2.0   # 첫 탐색 90도
            self.search_initialized = True
        else:
            angle_deg = base_angle_deg         # 이후 180도씩 좌우 회전

        signed_angle = angle_deg if self.search_left else -angle_deg
        self.search_left = not self.search_left

        rospy.loginfo(f"[SEARCH] 회전 탐색: {signed_angle:.1f}° 회전 중...")
        self._rotate(angle_deg=signed_angle, angular_speed=angular_speed)

        # 🔹 회전이 끝난 후에 타겟이 발견된 경우만 APPROACH로 진입
        if self.target_found:
            rospy.loginfo("[SEARCH] 회전 완료 후 타겟 발견 → APPROACH 모드 진입")

            try:
                if not hasattr(self, 'soundhandle'):
                    self.soundhandle = SoundClient()
                    rospy.sleep(0.2)
                self.soundhandle.play(SoundRequest.NEEDS_UNPLUGGING)
                rospy.loginfo("[SOUND] 출발 신호음 재생")
            except Exception as e:
                rospy.logwarn(f"[SOUND] 사운드 재생 실패: {e}")

            rospy.sleep(1.0)
            self.state = "APPROACH"

    # ----------------------------- 접근 모드 -----------------------------

    def approach_target(self):
        rospy.loginfo(f"타겟 접근 시작: 현재={self.current_position}, 타겟={self.target_position}")

        # 초기 yaw (deg) → 진행방향
        yaw_deg = rospy.get_param("/current_yaw_deg", 0.0)
        self.current_yaw = math.radians(yaw_deg)  # 내부는 라디안 사용
        rospy.loginfo(f"[YAW] 초기 카메라 방향: {yaw_deg:.1f}°")

        # 타겟 좌표 유효성
        if self.target_position[0] < 0 or self.target_position[1] < 0:
            rospy.logwarn(f"잘못된 목표: {self.target_position} → SEARCH 복귀")
            self.state = "SEARCH"
            self.target_found = False
            return

        # A* 경로
        start = discretize(self.current_position)
        goal  = discretize(self.target_position)
        rospy.loginfo(f"그리드: start={start}, goal={goal}")

        obstacles = [(0, 1), (2, 0), (1, 3)]  # TODO: 실제 장애물 연동
        grid = create_grid(obstacles, grid_size=(24, 12))

        path = astar(grid, start, goal)
        if not path or len(path) < 2:
            rospy.logwarn("경로 없음 → SEARCH 복귀")
            self.state = "SEARCH"
            self.target_found = False
            return

        # BEV 경로 전송
        try:
            cell_size = 0.5
            coords_str = ';'.join([f"{x*cell_size},{y*cell_size}" for (x, y) in path])
            self.path_sock.sendto(coords_str.encode('utf-8'), self.path_addr)
            rospy.loginfo(f"[BEV] 경로 {len(path)}노드 전송")
        except Exception as e:
            rospy.logwarn(f"[BEV] 전송 실패: {e}")

        rospy.loginfo(f"A* 완료: {len(path)} 스텝")

        # 이동 파라미터
        speed        = 0.15
        cell_size    = 0.5
        move_time    = cell_size / speed
        rotate_speed = 0.5

        # 초기 진행방향(그리드 단위)
        dir_vec = self.yaw_to_dir(yaw_deg)
        rospy.loginfo(f"[DIR] 초기방향(BEV): {dir_vec}")
        dir_vec = (-dir_vec[0], dir_vec[1])

        rate = rospy.Rate(20)
        for i in range(1, len(path)):
            current, next_p = path[i - 1], path[i]
            dx, dy = next_p[0] - current[0], next_p[1] - current[1]

            self.twist.linear.x  = 0.0
            self.twist.angular.z = 0.0

            cross = dir_vec[0]*dy - dir_vec[1]*dx
            if (dx, dy) == dir_vec:
                rospy.loginfo("직진")
                self.twist.linear.x = -speed

            elif (-dx, -dy) == dir_vec:
                rospy.loginfo("후진")
                self.twist.linear.x = speed

            elif cross < 0:
                rospy.loginfo("반시계 90° 후 직진")
                self._rotate(angle_deg=90.0, angular_speed=rotate_speed)
                self.twist.linear.x = speed

            elif cross > 0:
                rospy.loginfo("시계 90° 후 직진")
                self._rotate(angle_deg=-90.0, angular_speed=rotate_speed)
                self.twist.linear.x = speed
            else:
                rospy.logwarn(f"비정상 방향: dx={dx}, dy={dy}")
                continue

            # 진행방향 갱신
            dir_vec = (-dx, dy)
            self.current_position[0] += dir_vec[0] * 0.5
            self.current_position[1] += dir_vec[1] * 0.5

            # 한 셀 이동(이동 중 오도메트리로 self.current_position/ yaw 보정됨)
            start_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - start_time < move_time:
                # 근접 도달 판단(0.3 m)
                dist = math.hypot(self.target_position[0]-self.current_position[0],
                                  self.target_position[1]-self.current_position[1])
                if dist <= 0.3:
                    rospy.loginfo(f"타겟 {dist:.2f}m → 정지")
                    self.stop_robot()
                    rospy.sleep(0.5)

                    self.target_locked = False
                    self.locked_target_position = None

                    self.state = "SEARCH"
                    self.target_found = False
                    return
                self.cmd_vel_pub.publish(self.twist)
                rate.sleep()

            self.stop_robot()
            rospy.sleep(0.2)

        rospy.loginfo("목표 도달 → SEARCH 복귀")
        self.stop_robot()
        rospy.sleep(1.0)
        self.state = "SEARCH"
        self.target_found = False


    # ----------------------------- 회전(오도메트리 보정 포함) -----------------------------

    def _rotate(self, angle_deg, angular_speed):
        """
        오도메트리 기반 범용 회전
        - 회전 완료 후에만 종료
        """
        start_yaw = self.get_current_yaw()
        target_yaw = self._normalize_angle(start_yaw + math.radians(angle_deg))
        direction = 1.0 if angle_deg >= 0 else -1.0

        rospy.loginfo(f"[ROTATE] target={angle_deg}°, speed={angular_speed}rad/s")

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            current_yaw = self.get_current_yaw()
            remaining = self._angle_diff(target_yaw, current_yaw)

            if abs(remaining) < math.radians(1.0):  # 오차 ±1°
                break

            self.twist.linear.x = 0.0
            self.twist.angular.z = direction * abs(angular_speed)
            self.cmd_vel_pub.publish(self.twist)
            rate.sleep()

        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo(f"[ROTATE] rotation complete ({angle_deg}°)")


    # ----------------------------- 메인 루프 -----------------------------

    def run(self):
        rospy.loginfo("탐색 대기 중...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state == "SEARCH":
                self.search_mode()
            elif self.state == "APPROACH":
                self.approach_target()
            rate.sleep()


if __name__ == '__main__':
    controller = ServingRobotController()
    controller.run()
