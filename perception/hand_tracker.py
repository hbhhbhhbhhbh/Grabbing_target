import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math


class HandTracker:
    def __init__(
        self,
        model_path="hand_landmarker.task",
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    ):
        self.mp = mp
        self.num_hands = num_hands
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def detect(self, frame):
        """
        输入：BGR frame (OpenCV)
        输出：
        {
            "hand_found": bool,
            "landmarks": [(x, y), ...],   # 21 个点
            "center": (cx, cy) or None,
            "hand_open": bool or None,
            "hand_size": float or None,   # 改用关键点距离，更稳定（不受握拳影响）
            "hand_scale": float or None,  # 新增：手腕到中指根的距离
            "bbox": [x1, y1, x2, y2] or None
        }
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        hand_info = {
            "hand_found": False,
            "landmarks": [],
            "center": None,
            "hand_open": None,
            "hand_size": None,
            "hand_scale": None,
            "bbox": None
        }

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return hand_info

        landmarks = result.hand_landmarks[0]

        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        center = (sum(xs) // len(xs), sum(ys) // len(ys))

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        hand_width = max_x - min_x
        hand_height = max_y - min_y
        
        # ============================================
        # 【核心修改】改用关键点距离计算手部尺寸
        # ============================================
        # 计算手腕 (0) 到中指指根 (9) 的欧氏距离
        # 这个距离在手张开或握拳时变化较小，更适合深度判断
        wrist = points[0]
        middle_finger_base = points[9]
        hand_scale = math.sqrt(
            (wrist[0] - middle_finger_base[0])**2 + 
            (wrist[1] - middle_finger_base[1])**2
        )

        # hand_size 现在使用关键点距离，不再使用包围盒面积
        # 这样握拳时不会误判为"手变小了=变远了"
        hand_size = hand_scale

        hand_info["hand_found"] = True
        hand_info["landmarks"] = points
        hand_info["center"] = center
        hand_info["hand_open"] = self._is_hand_open(points)
        hand_info["hand_size"] = hand_size
        hand_info["hand_scale"] = hand_scale
        hand_info["bbox"] = [min_x, min_y, max_x, max_y]

        return hand_info

    def draw(self, frame, hand_info):
        if not hand_info["hand_found"]:
            return frame

        points = hand_info["landmarks"]

        # 画骨架连线
        for start_idx, end_idx in self.hand_connections:
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2) 

        # 画关键点
        for p in points:
            cv2.circle(frame, p, 4, (255, 0, 0), -1)

        # 画手中心
        cx, cy = hand_info["center"]
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

        # 画手包围框（调试前后距离很有用）
        if hand_info["bbox"] is not None:
            x1, y1, x2, y2 = hand_info["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # 显示手部状态信息
        cv2.putText(
            frame,
            f"Hand open: {hand_info['hand_open']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # 显示手部尺寸（关键点距离）
        if hand_info["hand_size"] is not None:
            cv2.putText(
                frame,
                f"Hand size: {hand_info['hand_size']:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
        
        # 显示 hand_scale（调试用，与 hand_size 相同）
        if hand_info["hand_scale"] is not None:
            cv2.putText(
                frame,
                f"Hand scale: {hand_info['hand_scale']:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        return frame

    def _is_hand_open(self, points):
        """
        简化版张手判断：
        比较四个手指尖 (8,12,16,20) 和手腕 (0) 的 y 坐标
        指尖平均位置更靠上，则视为张开
        """
        wrist_y = points[0][1]
        fingertip_ys = [points[8][1], points[12][1], points[16][1], points[20][1]]
        avg_tip_y = sum(fingertip_ys) / len(fingertip_ys)

        return avg_tip_y < wrist_y