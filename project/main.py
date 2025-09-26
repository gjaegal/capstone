# camera_stream.py에서 만든 RealSenseYOLOStreamer 사용
# ros_bridge.py의 targetPublisherROS 사용

import rospy
from camera_stream import RealSenseLocalizationStreamer
from ros_bridge import TargetPublisherROS
import multiprocessing
import bird_eye_view   

def on_detections_cb(dets):
    """
    dets : 리스트 [cls_name, depth_m, (x1, y1, x2, y2), (cx, cy), conf]
    이 호출은 ros_bridge.TargetPublisherROS로 라우팅 되므로 여기서는 별도 처리가 불필요
    """
    pass

def on_poses_cb(poses):
        """
    poses: list of dict(
        bbox=(x1,y1,x2,y2),
        keypoints=[(x,y) or (None,None)]*17,
        scores=[17] or None,
        depth_m=float or nan,
        conf=float
    )
    필요 시 ROS 토픽으로 퍼블리시하도록 확장 가능.
    """
    # 간단한 모니터링(필요 없으면 지워도 됨)
    # rospy.loginfo_throttle(1.0, f"[POSE] num_people={len(poses)}")

def main():
    # ROS 노드 초기화
    rospy.init_node('yolo_realsense_main', anonymous=True)

    # /target 퍼블리셔 브리지
    target_pub = TargetPublisherROS(node_name='yolo_realsense_publisher', topic='/target')

    # bird_eye_view를 별도 프로세스로 실행
    bev_process = multiprocessing.Process(target=bird_eye_view.main)
    bev_process.start()

    streamer = RealSenseLocalizationStreamer(
        yolo_det_weights='yolov8n.pt',
        yolo_pose_weights='yolov8n-pose.pt',
        tracker_max_age=5,
        show_windows=True,
        on_detections=target_pub.on_detections,  # /target으로 퍼블리시
        on_localization=target_pub.on_localization,
        on_poses=on_poses_cb                      # 필요 시 별도 토픽으로 확장
    )
    
    # streamer main loop
    # ROS는 퍼블리셔만 사용하므로 spin이 필요 없음
    try:
        streamer.run()
    except KeyboardInterrupt:
        pass
    finally:
        bev_process.terminate()   # 종료 시 bird_eye_view도 같이 종료
    
if __name__ == "__main__":
    main()
