
#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import tf
from astar import astar, create_grid, discretize
import socket

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
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.reached_pub = rospy.Publisher("/target_reached", Bool, queue_size = 1)

        # 상태/좌표
        self.state = "SEARCH"
        self.target_found = False
        self.current_position = [0.0, 0.0]
        self.current_yaw = 0.0
        self.target_position = [0.5, 0.5]
        self.current_dir = [1, 0]

        # 오도메트리 누적용
        self.last_odom_pos = None
        self.last_odom_yaw = None

        # 탐색 회전 방향 토글
        self.search_initialized = False
        self.search_left = True

        self.twist = Twist()
        rospy.loginfo("서빙 로봇 컨트롤러 시작 - YOLO+RealSense 모드")


    # ----------------------------- 콜백 -----------------------------

    def target_point_callback(self, msg):
        if not self.target_found:
            rospy.loginfo(f"##타겟 좌표##: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
            self.target_position = [msg.x, msg.y]
            self.target_found = True

    def current_point_callback(self, msg):
        if self.current_position != [msg.x, msg.y]:
            # rospy.loginfo(f"현재 좌표(vision): x={msg.x:.2f}, y={msg.y:.2f}, yaw(degree)={msg.z:.2f}")
            self.current_position = [msg.x, msg.y]
        self.current_dir = self.yaw_to_dir(msg.z)

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

    def rotate_right(self):
        '''시계 90도 회전'''
        self._rotate(angle_deg=90.0, angular_speed=0.5)

    def rotate_left(self):
        '''반시계 90도 회전'''
        self._rotate(angle_deg=-90.0, angular_speed=0.5)

    def rotate_back(self):
        " 시계 180도 회전"
        self._rotate(angle_deg=180.0, angular_speed=0.7)
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
        if   45 <= yaw_norm < 135:   return [0, 1]    # 북
        elif 135 <= yaw_norm < 225:  return [-1, 0]    # 서
        elif 225 <= yaw_norm < 315:  return [0, -1]   # 남
        else:                        return [1, 0]   # 동

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)


    # # ----------------------------- 탐색 모드 -----------------------------


    # ----------------------------- 접근 모드 -----------------------------

    def approach_target(self):
        dir_vec = self.current_dir
        rospy.loginfo(f"타겟 접근 시작: 현재={self.current_position}, 타겟={self.target_position}, 초기방향={dir_vec}")

        # 초기 yaw (deg) → 진행방향
        yaw_deg = rospy.get_param("/current_yaw_deg", 0.0)
        self.current_yaw = math.radians(yaw_deg)  # 내부는 라디안 사용
        rospy.loginfo(f"[YAW] 초기 YAW: {yaw_deg:.1f}°")

        # 타겟 좌표 유효성
        if self.target_position[0] < 0 or self.target_position[1] < 0:
            rospy.logwarn(f"잘못된 목표: {self.target_position} → SEARCH 복귀")
            return

        # A* 경로
        start = discretize(self.current_position)
        goal  = discretize(self.target_position)
        rospy.loginfo(f"그리드: start={start}, goal={goal}")

        obstacles = [(6, 3), (10, 5), (1, 3)]  # TODO: 실제 장애물 연동
        grid = create_grid(obstacles, grid_size=(24, 12))

        path = astar(grid, start, goal)
        if not path or len(path) < 2:
            rospy.logwarn("경로 없음 → SEARCH 복귀")
            return

        rospy.loginfo(f"[A*] 전체 경로: {path}")

        # 이동 파라미터
        speed        = 0.2
        cell_size    = 0.5
        move_time    = cell_size / speed
        rotate_speed = 0.5

        rate = rospy.Rate(20)
        for i in range(1, len(path)):
            current, next_p = path[i - 1], path[i]
            rospy.loginfo(f"[PATH] {current} -> {next_p} 이동")
            dx, dy = next_p[0] - current[0], next_p[1] - current[1]

            cross = dir_vec[0]*dy - dir_vec[1]*dx
            if [dx, dy] == dir_vec:
                rospy.loginfo("직진")
            elif cross < 0:
                rospy.loginfo("반시계 90° 후 직진")
                self.rotate_left()
            elif cross > 0:
                rospy.loginfo("시계 90° 후 직진")
                self.rotate_right()
            else:
                rospy.loginfo("180° 회전 후 직진")
                self.rotate_back()

            # 진행방향 갱신
            dir_vec = [dx, dy]
            self.current_dir = [dx, dy]

            # 한 셀 이동(이동 중 오도메트리로 self.current_position/ yaw 보정됨)
            self.twist.linear.x = speed
            self.twist.angular.z = 0.0

            start_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - start_time < move_time:
                # 근접 도달 판단(0.3 m)
                dist = math.hypot(self.target_position[0]-self.current_position[0],
                                  self.target_position[1]-self.current_position[1])
                if dist <= 0.1:
                    rospy.loginfo(f"타겟 {dist:.2f}m → 정지")
                    self.reached_pub.publish(True)
                    self.stop_robot()
                    rospy.sleep(0.5)
                    return
                self.cmd_vel_pub.publish(self.twist)
                rate.sleep()
            
            self.stop_robot()
            rospy.sleep(0.2)
            self.current_position[0] += dir_vec[0] * 0.5
            self.current_position[1] += dir_vec[1] * 0.5
            ################### 한 셀 이동 완료 ####################
            

        rospy.loginfo("목표 도달 → SEARCH 복귀")
        self.stop_robot()
        rospy.sleep(1.0)


    

    # ----------------------------- 메인 루프 -----------------------------

    def run(self):
        rospy.loginfo("탐색 대기 중...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # SEARCH MODE
            self.state = "SEARCH"
            if self.search_left:
                self.rotate_left()
                self.rotate_right()
                self.search_left = False
            else:
                self.rotate_right()
                self.rotate_left()
                self.search_left = True
            # self.search_left = not self.search_left

            rate.sleep()
            # 회전 동안 target을 찾았다면 APPROACH 모드로 전환
            if self.target_found == True:
                self.state = "APPROACH"
                self.approach_target()
                # 도착(혹은 error) 후 SEARCH 모드로 복귀
                self.state= "SEARCH"
                self.target_found = False
            rate.sleep()


if __name__ == '__main__':
    controller = ServingRobotController()
    controller.run()
