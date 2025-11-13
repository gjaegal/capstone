
class GestureDetector:
    def __init__(self, min_conf_pose=0.40):
        # 포즈 confidence threshold
        self.min_conf_pose = min_conf_pose

    def extract_absolute_keypoints(self, crop_keypoints, x1, y1):
        """
        YOLO pose keypoints (crop 기준) → 원본 이미지 절대 좌표로 변환
        crop_keypoints: (x, y, conf) 형태
        (x1, y1): crop의 좌측 상단 좌표
        """
        abs_kpts = []

        for (kx, ky, kc) in crop_keypoints:
            if kc < self.min_conf_pose:
                abs_kpts.append((None, None))
            else:
                abs_kpts.append((int(x1 + kx), int(y1 + ky)))
        
        return abs_kpts

    def is_hand_raised(self, abs_kpts):
        """
        절대좌표 keypoints → 손들기 판단
        head = 0번 keypoint
        left wrist = 9번
        right wrist = 10번
        """

        if len(abs_kpts) < 11:
            return False

        head = abs_kpts[0]
        left_wrist = abs_kpts[9]
        right_wrist = abs_kpts[10]

        if None in head:
            return False
        
        hx, hy = head

        # y값이 더 작아야 화면 상단 → 손이 머리보다 위
        if left_wrist is not None and None not in left_wrist:
            if left_wrist[1] < hy - 20:
                return True

        if right_wrist is not None and None not in right_wrist:
            if right_wrist[1] < hy - 20:
                return True

        return False
