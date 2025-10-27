#!/usr/bin/env python3

# ======== [Windows 테스트용 더미 rospy 모듈 추가] ========
try:
    import rospy
except ModuleNotFoundError:
    import types, time
    rospy = types.SimpleNamespace()
    rospy.loginfo = print
    rospy.logwarn = print
    rospy.sleep = time.sleep
    rospy.Time = types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_sec=lambda: time.time()))
    rospy.init_node = lambda name: print(f"[FAKE ROS] Node '{name}' initialized")
    rospy.Subscriber = lambda *a, **k: None
    rospy.Publisher = lambda *a, **k: types.SimpleNamespace(publish=lambda msg: None)
    rospy.spin = lambda: None
# =========================================================

# ====== [Windows 환경 테스트용 ROS 메시지 더미 클래스] ======
try:
    from geometry_msgs.msg import Twist, Point
    from vision_msgs.msg import Detection2DArray
except ModuleNotFoundError:
    import types
    class Twist:
        def __init__(self):
            self.linear = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
    class Point:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z
    class Detection2DArray:
        def __init__(self):
            self.detections = []
# ==========================================================



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
        self.state = "SEARCH"
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
        """YOLO + RealSense 기반 타겟 인식 콜백 (ID만 판별)"""
        if not msg.detections:
            return

        rospy.loginfo(f"타겟 감지 신호 수신: {len(msg.detections)}개")

        for detection in msg.detections:
            if not detection.results:           # 실제 카메라 환경에서 빈 result 있을 수 있어서 만든 안전장치
                continue

            result = detection.results[0]
            target_id = result.id

            rospy.loginfo(f"물체 감지 - ID: {target_id}")

            # 타겟 ID가 목표와 일치하면 이동 시작
            if target_id == self.target_id:
                self.target_found = True
                rospy.loginfo("타겟 인식 성공! A* 주행 모드로 전환")
                self.state = "APPROACH"
                return





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

        # # 1현재 좌표 → 격자 좌표 변환
        # start = discretize(self.current_position)
        # goal = discretize(self.target_position)
        # rospy.loginfo(f"그리드 현재 위치: {start}, 그리드 타겟 위치: {goal}")

        # # 2️격자 및 장애물 생성
        # obstacles = [(1,1), (1,2), (2,1), (2,2)]
        # grid = create_grid(obstacles, grid_size=(200,100))

        # # 3️A* 실행
        # path = astar(grid, start, goal)




        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

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
                rospy.loginfo("오른쪽 이동")
                self.twist.linear.x = speed
            elif dx == -1 and dy == 0:
                rospy.loginfo("왼쪽 이동")
                self.twist.linear.x = -speed

            elif dx == 0 and dy == 1:
                rospy.loginfo("위로 이동, 반시계 회전 후 직진")
                # 반시계 방향 90도 회전
                self.twist.angular.z = 0.5
                start_time = time.time()
                while time.time() - start_time < 1.6:
                    self.cmd_vel_pub.publish(self.twist)
                    time.sleep(0.1)
                
                self.stop_robot()
                rospy.sleep(0.3)
                # 회전 후 직진 (x축 기준)
                self.twist.angular.z = 0.0
                self.twist.linear.x = speed

            elif dx == 0 and dy == -1:
                rospy.loginfo("아래로 이동, 시계 회전 후 직진")
                # 시계 방향 90도 회전
                self.twist.angular.z = -0.5
                start_time = time.time()
                while time.time() - start_time < 1.6:
                    self.cmd_vel_pub.publish(self.twist)
                    time.sleep(0.1)

                self.stop_robot()
                rospy.sleep(0.3)
                # 회전 후 직진 (x축 기준)
                self.twist.angular.z = 0.0
                self.twist.linear.x = speed

            else:
                rospy.logwarn(f" 비정상적 이동 방향: dx={dx}, dy={dy}")
                continue

            # 실제 이동 (3초 동안 속도 명령 유지)
            start_time = time.time()
            while time.time() - start_time < move_time:
                self.cmd_vel_pub.publish(self.twist)
                time.sleep(0.1)

            # 한 칸 이동 후 정지
            self.stop_robot()
            rospy.sleep(0.2)
        # --- 여기까지 새로 추가된 핵심 이동 로직 ---

        # 도착 후 정지
        rospy.loginfo("목표 도착! 서빙 완료, 탐색 모드 복귀")
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



if __name__ == "__main__":
    import time

    print("\n[테스트 시작] A* 경로 주행 시뮬레이션\n")

    controller = ServingRobotController()

    # ① 테스트용 현재 좌표와 타겟 좌표 설정 (실제 좌표 단위: m)
    controller.current_position = (0.0, 0.0)
    controller.target_position = (0.12, 0.09)  # 약 12cm x 9cm 떨어진 목표

   # ② ROS 퍼블리셔 대신 print로 출력하도록 임시 함수 추가
    def fake_publish_twist(twist):
        print(f"[CMD] linear.x={twist.linear.x:.3f}, angular.z={twist.angular.z:.3f}")

    controller.cmd_vel_pub.publish = fake_publish_twist

    # ③ A* 이동 테스트 실행
    controller.state = "APPROACH"
    controller.approach_target()

    print("\n[테스트 종료] --- 로봇이 목표 지점까지 이동 완료 ---\n")
