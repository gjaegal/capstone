#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray



class ServingRobotController:
    def __init__(self):
        rospy.init_node('serving_robot_controller')
        
        # 로봇 제어 퍼블리셔
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1) # ('cmd_vel')를 이걸로 바꿈
        # YOLO+RealSense 감지 결과 구독
        self.target_sub = rospy.Subscriber('/target', Detection2DArray, self.target_callback)

        # 상태 변수들
        self.state = "SEARCH"
        self.target_id = 1
        self.target_found = False
        self.serving_complete = False
        self.current_target_depth = float('inf')  # 현재 타겟까지 거리
        self.obstacle_detected = False            # RealSense 기반 장애물 감지 여부

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

    def search_mode(self):
        """탐색 모드: 회전"""
        rospy.loginfo("탐색 중... 회전")
        if self.target_found:
            self.state = "APPROACH"
        else:
            if self.obstacle_detected:
                timer = 0
                while timer < 10:
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0.5
                    self.cmd_vel_pub.publish(self.twist)
                    timer += 1
            else:
                self.twist.linear.x = 0.02
                self.cmd_vel_pub.publish(self.twist)

    def approach_target(self):
        """타겟 접근"""
        rospy.loginfo("타겟 발견! 접근 중...")
        self.twist.linear.x = 0.2
        self.cmd_vel_pub.publish(self.twist)
        if self.obstacle_detected:
            rospy.loginfo("장애물 감지! 회피 중...")
            self.avoid_obstacle()
        elif self.current_target_depth < 0.5:
            self.state = "STOP"
            rospy.loginfo("타겟에 도착!")

    def avoid_obstacle(self):
        """장애물 회피 (Depth 기반)"""
        self.twist.angular.z = 0.8  # 회전하며 피하기
        self.twist.linear.x = -0.1  # 살짝 후진
        self.cmd_vel_pub.publish(self.twist)

    def stop_robot(self):
        """정지"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    def run(self):
        """메인 루프"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state == "SEARCH":
                self.search_mode()
            elif self.state == "APPROACH":
                self.approach_target()
            elif self.state == "STOP":
                self.stop_robot()
            rate.sleep()


if __name__ == '__main__':
    controller = ServingRobotController()
    controller.run()
