import rospy
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point, PointStamped

class TargetPublisherROS:
    """RealSenseYOLOStreamer의 on_detections 콜백을 받아 vision_msgs/Detection2DArray로 퍼블리시
    """
    def __init__(self, node_name='yolo_realsense_publisher', topic = '/target', queue_size=1):
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)
        self.pub = rospy.Publisher(topic, Detection2DArray, queue_size=queue_size)
        self.rate = rospy.Rate(10)
        self.current_pub = rospy.Publisher('/current_point', Point, queue_size=queue_size)
        self.target_pub = rospy.Publisher('/target_point', Point, queue_size=queue_size)
        
    def on_detections(self, dets):
        """
        dets: 다음 리스트 [cls_name, depth_m, (x1, y1, x2, y2), (cx, cy), conf]
        """
        msg = Detection2DArray()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        
        for cls_name, depth_m, (x1, y1, x2, y2), (cx, cy), conf in dets:
            det = Detection2D()
            hypo = ObjectHypothesisWithPose()
            if cls_name in ["chair", "remote", "mouse"]:
                hypo.id = 1
            else:
                hypo.id = 0
            hypo.score = conf
            
            hypo.pose.pose.position.x = 0.0
            hypo.pose.pose.position.y = 0.0
            hypo.pose.pose.position.z = float(depth_m)
            det.results.append(hypo)
            
            det.bbox.center.x = float(cx)
            det.bbox.center.y = float(cy)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            
            msg.detections.append(det)

        self.pub.publish(msg)
    
    def publish_point(self, P, location_type="current", yaw=None):
        """
        좌표 P: [x, y, z]
        location_type: "current" 로봇 위치, "target" 목표 위치
        """
        point = Point()
        point.x = P[0]
        point.y = P[1]
        point.z = P[2]
        # 현재 위치: /current_point 으로 publish
        if location_type=="current":
            self.current_pub.publish(point)
        # 타겟 위치: /target_point 으로 publish
        if location_type=="target":
            self.target_pub.publish(point)
        
        # 현재 방향: /current_yaw_deg 로 publish
        if yaw is not None:
            rospy.set_param("/current_yaw_deg", yaw)
