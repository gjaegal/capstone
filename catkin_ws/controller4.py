#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import tf
from astar import astar, create_grid, discretize
import socket

class ServingRobotController:
    def __init__(self):
        rospy.init_node('serving_robot_controller')
        
        # 로봇 제어 퍼블리셔
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
        self.current_point_sub = rospy.Subscriber('/current_point', Point, self.current_point_callback)
        self.target_point_sub = rospy.Subscriber('/target_point', Point, self.target_point_callback)


        # --- BEV 경로 전송용 UDP 소켓 ---
        self.path_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.path_addr = ('localhost', 12348)   # bird_eye_view.py에서 수신할 포트

        # --- BEV 장애물 전송용 UDP 소켓 ---
        self.obst_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.obst_addr = ('localhost', 12349)

        # 상태 변수
        self.state = "SEARCH"
        self.target_found = False
        self.current_position = (0.0, 0.0)
        self.target_position = (0.5, 0.5)

        self.twist = Twist()
        rospy.loginfo("서빙 로봇 컨트롤러 시작 - YOLO+RealSense 모드")


    # ----------------------------- 콜백 및 유틸리티 함수들-----------------------------

    def target_point_callback(self, msg):
        if self.target_found:
            return
            
        rospy.loginfo(f"타겟 좌표: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
        self.target_position = (msg.x, msg.y)
        self.target_found = True

    def current_point_callback(self, msg):
        if self.current_position != (msg.x, msg.y):
            rospy.loginfo(f"현재 좌표: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}")
            self.current_position = (msg.x, msg.y)

    def get_current_yaw(self):
        """
        /odom 토픽에서 현재 yaw(라디안)를 반환
        """
        try:
            odom_msg = rospy.wait_for_message("/odom", Odometry, timeout=1.0)
            orientation_q = odom_msg.pose.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([
                orientation_q.x,
                orientation_q.y,
                orientation_q.z,
                orientation_q.w
            ])
            return yaw
        except Exception as e:
            rospy.logwarn(f"[SEARCH] Failed to get yaw: {e}")
            return 0.0

    def _normalize_angle(self, angle):
        """[-pi, pi] 범위로 정규화"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, target, current):
        """현재 yaw와 목표 yaw의 차이 계산 ([-pi, pi] 범위 보정 포함)"""
        diff = self._normalize_angle(target - current)
        return diff
    # ----------------------------- 탐색 모드 -----------------------------

    # controller4.py의 search_mode 교체용(핵심만)
    def search_mode(self):
        if self.target_found:
            rospy.loginfo('[STATE] SEARCH -> APPROACH')
            self.state = "APPROACH"
            return

        base_angle_deg = 60.0    # 회전할 각도
        angular_speed = 0.5 # 각속도

        if not hasattr(self, 'search_initialized'):
            self.search_initialized = False
        if not hasattr(self, 'search_left'):
            self.search_left = True

        # 최초 실행이면 각도를 절반으로 줄이기
        if not self.search_initialized:
            angle_deg = base_angle_deg / 2.0   # 예: 45도
            self.search_initialized = True
        else:
            angle_deg = base_angle_deg         # 이후에는 정상 각도

        # 좌우 번갈아 회전
        if self.search_left:
            signed_angle = angle_deg
        else:
            signed_angle = -angle_deg
        self.search_left = not self.search_left

        rospy.loginfo(f"[SEARCH] rotating {angle_deg}°...")
        self._rotate(angle_deg = signed_angle, angular_speed = angular_speed)

    # ----------------------------- 접근 모드 -----------------------------

    def approach_target(self):
        """A* 경로를 grid 단위로 이동"""
        rospy.loginfo(f"타겟 접근 시작: 현재 위치={self.current_position}, 타겟={self.target_position}")

        # 카메라 yaw → 초기 진행방향 dir 설정
        yaw_deg = rospy.get_param("/current_yaw_deg", 0.0)
        self.current_yaw = yaw_deg
        rospy.loginfo(f"[YAW] 카메라 방향: {yaw_deg:.1f}°")

        # 유효 타겟 좌표 확인
        if self.target_position[0] < 0 or self.target_position[1] < 0:
            rospy.logwarn(f"잘못된 목표 좌표: {self.target_position}")
            self.state = "SEARCH"
            self.target_found = False
            return

        # 격자 변환 및 A* 경로탐색
        start = discretize(self.current_position)
        goal  = discretize(self.target_position)
        rospy.loginfo(f"그리드 좌표: start={start}, goal={goal}")

        obstacles = [(0, 1), (2, 0), (1, 3)]  # TODO: 실제 장애물 연동
        grid = create_grid(obstacles, grid_size=(24, 12))

        path = astar(grid, start, goal)
        if not path or len(path) < 2:
            rospy.logwarn("경로를 찾지 못했습니다. 탐색 모드로 복귀합니다.")
            self.state = "SEARCH"
            self.target_found = False
            return

        # BEV 경로 전송
        try:
            cell_size = 0.5  # m/셀
            coords_str = ';'.join([f"{x*cell_size},{y*cell_size}" for (x, y) in path])
            self.path_sock.sendto(coords_str.encode('utf-8'), self.path_addr)
            rospy.loginfo(f"[BEV] 경로 {len(path)}개 노드 전송 완료")

        # 장애물 전송
            obstacle_str = ';'.join([f"{x*cell_size},{y*cell_size}" for (x, y) in obstacles])
            self.obst_sock.sendto(obstacle_str.encode('utf-8'), self.obst_addr)
            rospy.loginfo(f"[BEV] 장애물 {len(obstacles)}개 전송 완료")



        except Exception as e:
            rospy.logwarn(f"[BEV] 경로 전송 실패: {e}")

        rospy.loginfo(f"A* 경로 탐색 완료: {len(path)} 스텝")

        # 이동 제어 파라미터
        speed         = 0.1    # m/s
        cell_size     = 0.5    # m
        move_time     = cell_size / speed
        rotate_speed  = 0.5    # rad/s

        # 초기 진행방향: yaw 기반 (북·동·남·서 중 하나)
        dir = self.yaw_to_dir(yaw_deg)
        rospy.loginfo(f"[DIR] 초기방향(BEV 기준) : {dir}")

        # 경로를 따라 이동
        rate = rospy.Rate(20)
        for i in range(1, len(path)):
            current, next_p = path[i - 1], path[i]
            dx, dy = next_p[0] - current[0], next_p[1] - current[1]

            # 회전/직진 결정
            self.twist.linear.x  = 0.0
            self.twist.angular.z = 0.0

            cross = dir[0]*dy - dir[1]*dx
            if (dx, dy) == dir:
                rospy.loginfo("직진")
                self.twist.linear.x = speed

            elif (-dx, -dy) == dir:
                rospy.loginfo("후진")
                self.twist.linear.x = -speed

            elif cross > 0:
                rospy.loginfo("반시계 90° 회전 후 직진")
                # 정확한 회전을 위해 오도메트리 기반 회전 사용
                self._rotate(angle_deg = 90.0, angular_speed = rotate_speed)
                self.twist.linear.x = speed

            elif cross < 0:
                rospy.loginfo("시계 90° 회전 후 직진")
                self._rotate(angle_deg = -90.0, angular_speed = rotate_speed)
                self.twist.linear.x = speed

            else:
                rospy.logwarn(f"비정상 방향: dx={dx}, dy={dy}")
                continue

            # 현재 진행방향 갱신
            dir = (dx, dy)

            # 셀 한 칸 이동
            start_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - start_time < move_time:
                # 타겟 근접 검사 (0.3 m 이내면 정지)
                dist_to_target = (
                    (self.target_position[0] - self.current_position[0]) ** 2 +
                    (self.target_position[1] - self.current_position[1]) ** 2
                ) ** 0.5

                if dist_to_target <= 0.3:
                    rospy.loginfo(f"타겟과 {dist_to_target:.2f}m 거리 → 정지")
                    self.stop_robot()
                    rospy.sleep(0.5)
                    self.state = "SEARCH"
                    self.target_found = False
                    return  # 즉시 종료

                self.cmd_vel_pub.publish(self.twist)
                rate.sleep()

            self.stop_robot()
            rospy.sleep(0.2)

        rospy.loginfo("목표 도달 → 탐색 모드 복귀")
        self.stop_robot()
        rospy.sleep(1.0)
        self.state = "SEARCH"
        self.target_found = False



    # ----------------------------- 보조 함수 -----------------------------

    # def _rotate(self, angular_speed, duration):
    #     """지정된 속도로 회전"""
    #     self.twist.angular.z = angular_speed
    #     self.twist.linear.x = 0.0
    #     start_time = rospy.Time.now().to_sec()
    #     while rospy.Time.now().to_sec() - start_time < duration:
    #         self.cmd_vel_pub.publish(self.twist)
    #     self.stop_robot()
    #     rospy.sleep(0.3)

    def _rotate(self, angle_deg, angular_speed):
        """
        오도메트리 기반 범용 회전 함수
        angle_deg: 회전 각도 (+는 좌회전, -는 우회전)
        angular_speed: 회전 속도 (rad/s)
        """
        # 현재 yaw 가져오기
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

        # 회전 종료 → 정지
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo(f"[ROTATE] rotation complete (Δθ={math.degrees(self._angle_diff(target_yaw, current_yaw)):.1f}°)")


    def yaw_to_dir(self, yaw_deg):
        yaw_norm = yaw_deg % 360.0

        if 45<= yaw_norm < 135 : # 동쪽
            return (1,0)
        elif 135 <= yaw_norm < 225:
            return (0, -1) # 남쪽
        elif 225<= yaw_norm < 315: # 서쪽
            return(-1, 0)
        else: return (0,1) # 북쪽

    def stop_robot(self):
        """정지"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    def run(self):
        """메인 루프"""
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
