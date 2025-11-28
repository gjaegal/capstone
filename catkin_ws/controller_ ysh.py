#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import tf
from astar import astar, create_grid, discretize

# ---- 간단한 2D 벡터/회전 유틸 (필요 시 확장용) ----
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
        rospy.Subscriber('/current_point', Point, self.current_point_callback)
        rospy.Subscriber('/target_point',  Point, self.target_point_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.reached_pub = rospy.Publisher("/target_reached", Bool, queue_size=1)

        # 상태/좌표
        self.state = "SEARCH"           # SEARCH → APPROACH → WAIT
        self.target_found = False
        self.current_position = [0.0, 0.0]
        self.current_yaw = 0.0          # rad
        self.target_position = [0.5, 0.5]
        self.current_dir = [1, 0]       # 격자 기준 진행 방향

        # 오도메트리 누적용 (원하면 여기에 쌓아서 보정 사용 가능)
        self.last_odom_pos = None
        self.last_odom_yaw = None

        # 탐색 회전 방향 토글 (좌 ↔ 우)
        self.search_left = True

        # WAIT 모드 파라미터
        self.wait_after_approach = 10.0  # 목표 도달 후 대기 시간(초)

        self.twist = Twist()
        rospy.loginfo("서빙 로봇 컨트롤러 시작 - YOLO+RealSense 모드")

        self.first_search_rotation = True

    # ----------------------------- 콜백 -----------------------------

    def target_point_callback(self, msg):
        # 새로운 타겟 들어왔을 때만 세팅
        if not self.target_found:
            rospy.loginfo(f"##타겟 좌표##: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
            self.target_position = [msg.x, msg.y]
            self.target_found = True

    def current_point_callback(self, msg):
        # vision 기반 현재 좌표
        self.current_position = [msg.x, msg.y]
        self.current_dir = self.yaw_to_dir(msg.z)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        odom_now = (p.x, p.y)
        self.current_yaw = yaw

        if self.last_odom_pos is None:
            self.last_odom_pos = odom_now
            self.last_odom_yaw = yaw

    # ----------------------------- 속도 램핑 -----------------------------

    def ramp_speed(self, target_linear=0.0, target_angular=0.0,
                   step_linear=0.015, step_angular=0.02, rate_hz=30):
        """
        선속도/각속도를 부드럽게 목표치까지 증가/감소시키는 함수
        - 현재 self.twist 기준으로 target까지 선형 interpolation
        - 호출이 끝날 때까지 blocking
        """
        rate = rospy.Rate(rate_hz)

        current_linear = self.twist.linear.x
        current_angular = self.twist.angular.z

        while not rospy.is_shutdown():
            dl = target_linear - current_linear
            da = target_angular - current_angular

            if abs(dl) < 0.01 and abs(da) < 0.01:
                break

            # 선속도 업데이트
            if dl > 0:
                current_linear += min(step_linear, dl)
            else:
                current_linear += max(-step_linear, dl)

            # 각속도 업데이트
            if da > 0:
                current_angular += min(step_angular, da)
            else:
                current_angular += max(-step_angular, da)

            self.twist.linear.x = current_linear
            self.twist.angular.z = current_angular
            self.cmd_vel_pub.publish(self.twist)
            rate.sleep()

        # 최종 보정
        self.twist.linear.x = target_linear
        self.twist.angular.z = target_angular
        self.cmd_vel_pub.publish(self.twist)

    # ----------------------------- 회전(오도메트리 기반) -----------------------------

    def _rotate(self, angle_deg, angular_speed):
        """
        오도메트리 기반 회전
        - angle_deg 만큼 회전 (양수: 시계, 음수: 반시계 or 그 반대 – 여기선 get_current_yaw 기준)
        - angular_speed(rad/s)까지 ramp-up 후, yaw 오차 ±1° 이내가 되면 ramp-down
        """

        start_yaw = self.get_current_yaw()
        target_yaw = self._normalize_angle(start_yaw + math.radians(angle_deg))
        direction = 1.0 if angle_deg >= 0 else -1.0
        target_w = direction * abs(angular_speed)

        rospy.loginfo(f"[ROTATE] target={angle_deg}°, speed={target_w}")

        # 즉시 회전 시작
        self.twist.linear.x = 0.0
        self.twist.angular.z = target_w
        self.cmd_vel_pub.publish(self.twist)

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            current_yaw = self.get_current_yaw()
            remaining = self._angle_diff(target_yaw, current_yaw)
            if abs(remaining) < math.radians(1.0):
                break
            self.cmd_vel_pub.publish(self.twist)
            rate.sleep()

        # 즉시 회전 정지
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

        rospy.loginfo(f"[ROTATE] complete ({angle_deg} deg)")

    def rotate_right(self):
        """시계 90도 회전"""
        self._rotate(angle_deg=90.0, angular_speed=0.5)

    def rotate_left(self):
        """반시계 90도 회전"""
        self._rotate(angle_deg=-90.0, angular_speed=0.5)

    def rotate_back(self):
        """180도 회전"""
        self._rotate(angle_deg=180.0, angular_speed=0.7)

    # ----------------------------- 유틸 -----------------------------

    def get_current_yaw(self):
        try:
            odom_msg = rospy.wait_for_message("/odom", Odometry, timeout=1.0)
            q = odom_msg.pose.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return yaw
        except Exception as e:
            rospy.logwarn(f"[YAW] Failed to get yaw: {e}")
            return self.current_yaw

    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, target, current):
        return self._normalize_angle(target - current)

    def yaw_to_dir(self, yaw_deg):
        yaw_norm = yaw_deg % 360.0
        if   45 <= yaw_norm < 135:   return [0, 1]   # 북
        elif 135 <= yaw_norm < 225:  return [-1, 0]  # 서
        elif 225 <= yaw_norm < 315:  return [0, -1]  # 남
        else:                        return [1, 0]   # 동

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    # ----------------------------- APPROACH 모드 -----------------------------

    def approach_target(self):
        """
        A* 경로를 따라 타겟으로 접근.
        성공 시 True, 실패/취소 시 False 반환.
        """
        dir_vec = self.current_dir
        rospy.loginfo(f"[APPROACH] 시작: 현재={self.current_position}, 타겟={self.target_position}, dir={dir_vec}")

        # 초기 yaw(deg)를 vision에서 가져올 수도 있지만, 여기선 그냥 current_dir로만 사용
        yaw_deg = rospy.get_param("/current_yaw_deg", 0.0)
        self.current_yaw = math.radians(yaw_deg)
        rospy.loginfo(f"[YAW] 초기 yaw={yaw_deg:.1f} deg")

        # 타겟 좌표 유효성 검사
        if self.target_position[0] < 0 or self.target_position[1] < 0:
            rospy.logwarn(f"[APPROACH] 잘못된 목표: {self.target_position} → 실패")
            return False

        # A* 경로 생성
        start = discretize(self.current_position)
        goal  = discretize(self.target_position)
        rospy.loginfo(f"[A*] grid start={start}, goal={goal}")

        # TODO: 실제 장애물 연동
        obstacles = [(6, 3), (10, 5), (1, 3)]
        grid = create_grid(obstacles, grid_size=(24, 12))

        path = astar(grid, start, goal)
        if not path or len(path) < 2:
            rospy.logwarn("[A*] 경로 없음 → APPROACH 실패")
            return False

        rospy.loginfo(f"[A*] 경로: {path}")

        # 이동 파라미터
        speed          = 0.2
        cell_size      = 0.5
        move_time      = cell_size / speed
        dist_threshold = 0.1

        rate = rospy.Rate(20)

        for i in range(1, len(path)):
            current, nxt = path[i-1], path[i]
            rospy.loginfo(f"[PATH] {current} → {nxt}")

            dx = nxt[0] - current[0]
            dy = nxt[1] - current[1]

            # 회전 방향 결정 (cross product)
            cross = dir_vec[0]*dy - dir_vec[1]*dx
            if [dx, dy] == dir_vec:
                rospy.loginfo("[DIR] 직진 유지")
            elif cross < 0:
                rospy.loginfo("[DIR] 반시계 90°")
                self.rotate_left()
            elif cross > 0:
                rospy.loginfo("[DIR] 시계 90°")
                self.rotate_right()
            else:
                rospy.loginfo("[DIR] 180° 회전")
                self.rotate_back()

            # 진행 방향 갱신
            dir_vec = [dx, dy]
            self.current_dir = [dx, dy]

            # 한 셀 전진 (부드러운 가속)
            self.ramp_speed(target_linear=speed,
                            target_angular=0.0,
                            step_linear=0.02,
                            step_angular=0.02)

            start_t = rospy.Time.now().to_sec()
            while not rospy.is_shutdown():
                elapsed = rospy.Time.now().to_sec() - start_t
                if elapsed >= move_time:
                    break

                # 근접 도달 판단
                dist = math.hypot(self.target_position[0] - self.current_position[0],
                                  self.target_position[1] - self.current_position[1])
                if dist <= dist_threshold:
                    rospy.loginfo(f"[APPROACH] 타겟 {dist:.3f}m → 도달")
                    # 부드러운 감속 + 도달 처리
                    self.ramp_speed(target_linear=0.0,
                                    target_angular=0.0,
                                    step_linear=0.03,
                                    step_angular=0.03)
                    self.reached_pub.publish(True)
                    self.target_found = False
                    rospy.sleep(0.5)
                    return True

                self.cmd_vel_pub.publish(self.twist)
                rate.sleep()

            # 셀 이동 종료 → 감속
            self.ramp_speed(target_linear=0.0,
                            target_angular=0.0,
                            step_linear=0.03,
                            step_angular=0.03)
            rospy.sleep(0.2)

            # 격자 기준 위치 업데이트 (rough)
            self.current_position[0] += dir_vec[0] * cell_size
            self.current_position[1] += dir_vec[1] * cell_size

        # 여기까지 왔으면 path 끝까지 간 것 → 도달로 간주
        rospy.loginfo("[APPROACH] 경로 끝까지 이동 → 타겟 도달 처리")
        self.reached_pub.publish(True)
        self.target_found = False
        self.stop_robot()
        rospy.sleep(0.5)
        return True

    # ----------------------------- WAIT 모드 -----------------------------

    def wait_mode(self):
        rospy.loginfo(f"[WAIT] {self.wait_after_approach:.1f}s 대기...")
        self.stop_robot()
        rospy.sleep(self.wait_after_approach)
        rospy.loginfo("[WAIT] 종료 → SEARCH 복귀")

    # ----------------------------- 메인 루프 (상태머신) -----------------------------

    def run(self):
        rospy.loginfo("[MAIN] 탐색 시작")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            # -------- SEARCH MODE --------
            if self.state == "SEARCH":
                # 타겟 이미 발견 ⇒ 바로 APPROACH
                if self.target_found:
                    rospy.loginfo("[STATE] SEARCH→APPROACH (타겟 감지)")
                    self.state = "APPROACH"
                    continue

                rospy.loginfo("[STATE] SEARCH - 회전하며 탐색중")
                # --- 수정 후 SEARCH 회전 로직 ---
                if self.first_search_rotation:
                    rospy.loginfo("[SEARCH] 최초 실행 → 90도 회전")
                    self.rotate_left()
                    self.first_search_rotation = False
                else:
                    # 좌우 번갈아 180도씩 회전
                    if not self.search_left:
                        rospy.loginfo("[SEARCH] 180도 왼쪽 회전")
                        self.rotate_left()
                        self.rotate_left()
                    else:
                        rospy.loginfo("[SEARCH] 180도 오른쪽 회전")
                        self.rotate_right()
                        self.rotate_right()

                    # 다음엔 반대 방향으로 회전
                    self.search_left = not self.search_left
                rate.sleep()
                continue

            # -------- APPROACH MODE --------
            elif self.state == "APPROACH":
                rospy.loginfo("[STATE] APPROACH")
                success = self.approach_target()

                if success:
                    rospy.loginfo("[STATE] APPROACH 성공 → WAIT")
                    self.state = "WAIT"
                else:
                    rospy.loginfo("[STATE] APPROACH 실패 → SEARCH")
                    self.state = "SEARCH"
                continue

            # -------- WAIT MODE --------
            elif self.state == "WAIT":
                rospy.loginfo("[STATE] WAIT")
                self.wait_mode()
                self.state = "SEARCH"
                continue


if __name__ == '__main__':
    controller = ServingRobotController()
    controller.run()
