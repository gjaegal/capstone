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
        self.target_sub = rospy.Subscriber('/target', Detection2DArray, self.target_callback)
        self.current_point_sub = rospy.Subscriber('/current_point', Point, self.current_point_callback)
        self.target_point_sub = rospy.Subscriber('/target_point', Point, self.target_point_callback)

        # 상태 변수들
        self.state = "APPROACH"
        self.target_id = 1
        self.target_found = False
        self.serving_complete = False
        self.current_target_depth = float('inf')  # 현재 타겟까지 거리
        self.obstacle_detected = False            # RealSense 기반 장애물 감지 여부
        self.current_position = (2.0, 2.0)        # 현재 로봇 위치 (x, y)
        self.target_position = (0.0, 0.0)         # 목표 위치 (x, y)
        self.prev_grid_sg = None # discretized grid에서의 (현재 xy, goal xy)

        self.twist = Twist()

        self.timer = 0

        rospy.loginfo("서빙 로봇 컨트롤러 시작 - YOLO+RealSense 모드")

    def target_callback(self, msg):
        """YOLO + RealSense 결과 처리"""
        if msg.detections:
            closest_target = None
            min_depth = float('inf')

            rospy.loginfo(f"콜백 감지, 감지 수: {len(msg.detections)}")

            for detection in msg.detections:
                if detection.results and detection.results[0].score >= 0.4:
                    result = detection.results[0]
                    print("물체 감지!\t ID : ", result.id)

                    depth = result.pose.pose.position.z

                    if result.id == self.target_id:
                        self.target_found = True
                        self.current_target_depth = depth

                    if depth < min_depth:
                        min_depth = depth


            # 일정 거리 이내면 장애물로 간주
            if min_depth < 0.4:  # 40cm 이내
                self.obstacle_detected = True
            else:
                self.obstacle_detected = False

            rospy.loginfo(f"타겟 발견! 거리: {min_depth:.2f}m | 장애물 여부: {self.obstacle_detected}")



    def target_point_callback(self, msg):
        rospy.loginfo(f"타겟 좌표: x={msg.x}, y={msg.y}, z={msg.z}")
        self.target_position = (msg.x, msg.y)

    def current_point_callback(self, msg):
        rospy.loginfo(f"현재 좌표: x={msg.x}, y={msg.y}, z={msg.z}")
        self.current_position = (msg.x, msg.y)


    def search_mode(self):
        """탐색 모드"""
        # rospy.loginfo("탐색 중... ")
        if self.target_found:
            self.state = "APPROACH"
            return
        
        if self.obstacle_detected:
            self.avoid_obstacle()
        else:
            self.twist.linear.x = 0.01
            self.cmd_vel_pub.publish(self.twist)


    def approach_target(self):
        """A* 경로를 따라 상하좌우로 이동"""
        rospy.loginfo(f"타겟 접근 시작! 현재 위치: {self.current_position}, 타겟 위치: {self.target_position}")

        # 1현재 좌표 → 격자 좌표 변환
        start = discretize(self.current_position)
        goal = discretize(self.target_position)
        rospy.loginfo(f"그리드 현재 위치: {start}, 그리드 타겟 위치: {goal}")

        # 2️격자 및 장애물 생성
        obstacles = [(1,1), (1,2), (2,1), (2,2)]
        grid = create_grid(obstacles, grid_size=(200,100))

        # 3️A* 실행
        path = astar(grid, start, goal)
        if not path:
            rospy.logwarn("경로를 찾지 못했습니다. 탐색으로 복귀합니다.")
            self.state = "SEARCH"
            return

        rospy.loginfo(f"A* 경로 길이: {len(path)}")
        for p in path:
            rospy.loginfo(f"  {p}")

        # ---여기부터 새로 추가된 핵심 이동 로직 ---
        speed = 0.01        # 0.01 m/s (1cm/s)
        cell_size = 0.03    # 한 칸 = 0.03 m (3cm)
        move_time = cell_size / speed  # 3초

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
            if dx == 1 and dy == 0:
                rospy.loginfo("➡️ 오른쪽 이동")
                self.twist.linear.x = speed
            elif dx == -1 and dy == 0:
                rospy.loginfo("⬅️ 왼쪽 이동")
                self.twist.linear.x = -speed
            elif dx == 0 and dy == 1:
                rospy.loginfo("⬆️ 위로 이동")
                self.twist.linear.y = speed
            elif dx == 0 and dy == -1:
                rospy.loginfo("⬇️ 아래로 이동")
                self.twist.linear.y = -speed
            else:
                rospy.logwarn(f"⚠️ 비정상적 이동 방향: dx={dx}, dy={dy}")
                continue

            # 실제 이동 (3초 동안 속도 명령 유지)
            start_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - start_time < move_time:
                self.cmd_vel_pub.publish(self.twist)

            # 한 칸 이동 후 정지
            self.stop_robot()
            rospy.sleep(0.2)
        # --- 여기까지 새로 추가된 핵심 이동 로직 ---

        # 도착 후 정지
        rospy.loginfo("✅ 목표 도착! 서빙 완료 → 탐색 모드 복귀")
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
        rospy.spin()     # ROS 이벤트 루프로 진입함 콜백 큐를 비우며 구독 중인 토픽의 콜백들을 반복 처리함



if __name__ == '__main__':
    controller = ServingRobotController()
    controller.run()
