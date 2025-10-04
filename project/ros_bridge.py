import rospy
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point

class TargetPublisherROS:
    """RealSenseYOLOStreamer의 on_detections 콜백을 받아 vision_msgs/Detection2DArray로 퍼블리시
    """
    def __init__(self, node_name='yolo_realsense_publisher', topic = '/target', queue_size=1):
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)
        self.pub = rospy.Publisher(topic, Detection2DArray, queue_size=queue_size)
        self.target_pub = rospy.Publisher('/target_xyz', Point, queue_size=queue_size)
        self.current_pub = rospy.Publisher('/current_xyz', Point, queue_size=queue_size)
        
    def on_detections(self, dets, Pw=None):
        """
        dets: 다음 리스트 [cls_name, depth_m, (x1, y1, x2, y2), (cx, cy), conf]
        Pw: target (x,y,z) in world coordinates
        """
        msg = Detection2DArray()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()

        target_point = Point()
        
        for cls_name, depth_m, (x1, y1, x2, y2), (cx, cy), conf in dets:
            det = Detection2D()
            hypo = ObjectHypothesisWithPose()
            if cls_name == "mouse" :
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

        for (x, y, z) in Pw:
            target_point.x = x
            target_point.y = y
            target_point.z = z

        self.pub.publish(msg)
        self.target_pub.publish(target_point)
    
    def on_localization(self, x, y, z):
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        self.current_pub.publish(point)
        
