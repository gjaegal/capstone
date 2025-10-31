#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from astar import astar, create_grid, discretize

class ServingRobotController:
    def __init__(self):
        rospy.init_node('serving_robot_controller')
        
        # 로봇 제어 퍼블리셔
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1) # ('cmd_vel')를 이걸로 바꿈
        # YOLO+RealSense 감지 결과 구독
        # self.target_sub = rospy.Subscriber('/target', Detection2DArray, self.target_callback)
        self.current_point_sub = rospy.Subscriber('/current_point', Point, self.current_point_callback)
        self.target_point_sub = rospy.Subscriber('/target_point', Point, self.target_point_callback)

        # 상태 변수들
        self.state = "SEARCH"
        self.target_id = 0

        self.target_found = False
        self.serving_complete = False
        self.current_target_depth = float('inf')  # 현재 타겟까지 거리
        self.obstacle_detected = False            # RealSense 기반 장애물 감지 여부
        self.current_position = (0.0, 0.0)        # 현재 로봇 위치 (x, y)
        self.target_position = (0.5, 0.5)         # 목표 위치 (x, y)
        self.prev_grid_sg = None # discretized grid에서의 (현재 xy, goal xy)

        self.twist = Twist()

        self.timer = 0

        rospy.loginfo("서빙 로봇 컨트롤러 시작 - YOLO+RealSense 모드")


    def target_point_callback(self, msg):
        rospy.loginfo(f"타겟 좌표: x={msg.x}, y={msg.y}, z={msg.z}")
        self.target_position = (msg.x, msg.y)
        self.target_found = True

    def current_point_callback(self, msg):
        if self.current_position != (msg.x, msg.y):
            rospy.loginfo(f"현재 좌표: x={msg.x}, y={msg.y}, z={msg.z}")
            self.current_position = (msg.x, msg.y)


    def search_mode(self):
        """탐색 모드 - 전후진 반복"""
        # 즉시 상태 변경 확인
        if self.target_found:
            rospy.loginfo('[STATE] SEARCH -> APPROACH')
            self.state = "APPROACH"
            return
        
        #if self.obstacle_detected:
        #    self.avoid_obstacle()
        #    return
        
        # 전후진 패턴
        if not hasattr(self, 'search_start_time'):
            self.search_start_time = rospy.Time.now()
            self.search_forward = True
        
        elapsed = (rospy.Time.now() - self.search_start_time).to_sec()
        
        # 3초마다 앞뒤 변경
        if elapsed > 3.0:
            self.search_forward = not self.search_forward
            self.search_start_time = rospy.Time.now()
        
        # 전진 또는 후진
        if self.search_forward:
            self.twist.linear.x = 0.1  # 전진
        else:
            self.twist.linear.x = -0.1  # 후진
        
        self.cmd_vel_pub.publish(self.twist)


    def approach_target(self):
        """A* 경로를 따라 상하좌우로 이동"""
        rospy.loginfo(f"타겟 접근 시작! 현재 위치: {self.current_position}, 타겟 위치: {self.target_position}")

        if self.target_position[0] < 0 or self.target_position[1] < 0:
            rospy.logwarn(f"잘못된 목표 좌표입니다: {self.target}")
            self.state = "SEARCH"
            self.target_found = False
            return

        # 1현재 좌표 → 격자 좌표 변환
        start = discretize(self.current_position)
        goal = discretize(self.target_position)
        rospy.loginfo(f"그리드 현재 위치: {start}, 그리드 타겟 위치: {goal}")

        # 2️격자 및 장애물 생성
        obstacles = [(0, 1), (2, 0), (1,3)]
        grid = create_grid(obstacles, grid_size=(24,12)) # 6m/3m grid 크기는 12m, 6m

        # 3️A* 실행
        path = astar(grid, start, goal)
        if not path:
            rospy.logwarn("경로를 찾지 못했습니다. 탐색으로 복귀합니다.")
            self.state = "SEARCH"
            self.target_found = False
            return

        rospy.loginfo(f"A* 경로 길이: {len(path)}")
        for p in path:
            rospy.loginfo(f"  {p}")

        # ---여기부터 새로 추가된 핵심 이동 로직 ---
        speed = 0.1        # 0.1 m/s (10cm/s)
        cell_size = 0.5    # 한 칸 = 0.5 m (50cm)
        move_time = cell_size / speed  # 5초
        rotate_speed = 0.5 # 회전 각속도 0.5 rad/s

        dir = (1, 0) # 초기 바라보는 방향: x축 방향

        for i in range(1, len(path)):
            current = path[i-1]
            next_p = path[i]
            dx = next_p[0] - current[0]
            dy = next_p[1] - current[1]

            # 초기화
            self.twist.linear.x = 0.0
            self.twist.linear.y = 0.0
            self.twist.angular.z = 0.0

            # 상하좌우 판단
            cross = dir[0]*dy - dir[1]*dx
            if (dx, dy) == dir:
                rospy.loginfo("직진")
                self.twist.linear.x = speed
            elif (-dx, -dy) == dir:
                rospy.loginfo("뒤로 직진")
                self.twist.linear.x = -speed

            elif cross > 0:
                rospy.loginfo("반시계 회전 후 직진")
                # 반시계 방향 90도 회전
                self.twist.angular.z = rotate_speed
                start_time = rospy.Time.now().to_sec()
                while rospy.Time.now().to_sec() - start_time < 5.2:
                    self.cmd_vel_pub.publish(self.twist)
                
                self.stop_robot()
                rospy.sleep(0.3)
                # 회전 후 직진 (x축 기준)
                self.twist.angular.z = 0.0
                self.twist.linear.x = speed

            elif cross < 0:
                rospy.loginfo("시계 회전 후 직진")
                # 시계 방향 90도 회전
                self.twist.angular.z = -rotate_speed
                start_time = rospy.Time.now().to_sec()
                while rospy.Time.now().to_sec() - start_time < 5.2: # TODO: 5.2초에서 90도 조금 넘게 회전
                    self.cmd_vel_pub.publish(self.twist)

                self.stop_robot()
                rospy.sleep(0.3)
                # 회전 후 직진 (x축 기준) 
                self.twist.angular.z = 0.0
                self.twist.linear.x = speed

            else:
                rospy.logwarn(f" 비정상적 이동 방향: dx={dx}, dy={dy}")
                continue
            
            # 바라보는 방향 업데이트
            dir = (dx, dy)
            
            # 직진 이동 (5초 동안 속도 명령 유지) TODO: 대략 38cm 이동, 12cm 오차
            start_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - start_time < move_time+1:
                self.cmd_vel_pub.publish(self.twist)

            # 한 칸 이동 후 정지
            self.stop_robot()
            rospy.sleep(0.2)
        # --- 여기까지 새로 추가된 핵심 이동 로직 ---

        # 도착 후 정지
        rospy.loginfo("도착-> 탐색 모드 복귀")
        self.stop_robot()
        rospy.sleep(1.0)
        self.state = "SEARCH"
        self.target_found = False

    def stop_robot(self):
        """정지"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    def run(self):
        """기본 루프 (대기만 함)"""
        rospy.loginfo("타겟 감지 대기 중...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state == "SEARCH":
                self.search_mode()
            if self.state == "APPROACH":
                self.approach_target()
            rate.sleep()



if __name__ == '__main__':
    controller = ServingRobotController()
    controller.run()
