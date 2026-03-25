import time
import math
from collections import deque

class OutputFormatter:
    def __init__(self, smooth_window=5):
        self.smooth_window = smooth_window
        # 物体中心平滑缓冲区
        self.object_center_buffer = deque(maxlen=smooth_window)
        # 物体面积平滑缓冲区
        self.object_area_buffer = deque(maxlen=smooth_window)
        # 物体线性尺寸平滑缓冲区（新增）
        self.object_size_buffer = deque(maxlen=smooth_window)
        # 手部尺寸平滑缓冲区（新增）
        self.hand_size_buffer = deque(maxlen=smooth_window)

    def _smooth_center(self, center):
        """平滑物体中心坐标"""
        if center is None:
            self.object_center_buffer.clear()
            return None

        self.object_center_buffer.append(center)

        avg_x = int(sum(c[0] for c in self.object_center_buffer) / len(self.object_center_buffer))
        avg_y = int(sum(c[1] for c in self.object_center_buffer) / len(self.object_center_buffer))

        return [avg_x, avg_y]

    def _smooth_area(self, area):
        """平滑物体面积"""
        if area is None or area <= 0:
            self.object_area_buffer.clear()
            return None

        self.object_area_buffer.append(area)
        return float(sum(self.object_area_buffer) / len(self.object_area_buffer))

    def _smooth_size(self, size):
        """平滑线性尺寸（新增）"""
        if size is None or size <= 0:
            self.object_size_buffer.clear()
            self.hand_size_buffer.clear()
            return None

        self.object_size_buffer.append(size)
        return float(sum(self.object_size_buffer) / len(self.object_size_buffer))

    def _smooth_hand_size(self, hand_size):
        """平滑手部尺寸（新增）"""
        if hand_size is None or hand_size <= 0:
            self.hand_size_buffer.clear()
            return None

        self.hand_size_buffer.append(hand_size)
        return float(sum(self.hand_size_buffer) / len(self.hand_size_buffer))

    def format_output(self, frame, detections, hand_info, preferred_label=None):
        """
        整合所有感知数据，输出统一格式给 guidance_controller
        """
        h, w, _ = frame.shape

        # 1. 选择最佳物体
        selected_object = None

        if preferred_label is not None:
            filtered = [d for d in detections if d["label"] == preferred_label]
            if filtered:
                selected_object = max(filtered, key=lambda x: x["area"])

        if selected_object is None and len(detections) > 0:
            selected_object = max(detections, key=lambda x: x["area"])

        # 2. 提取原始数据
        raw_center = selected_object["center"] if selected_object else None
        raw_area = selected_object["area"] if selected_object else None
        raw_object_size = selected_object.get("object_size") if selected_object else None
        raw_hand_size = hand_info.get("hand_size")

        # 3. 应用平滑处理
        smooth_center = self._smooth_center(raw_center) if raw_center is not None else None
        smooth_area = self._smooth_area(raw_area)
        smooth_object_size = self._smooth_size(raw_object_size)
        smooth_hand_size = self._smooth_hand_size(raw_hand_size)

        # 4. 构建输出字典（修复所有键名空格问题）
        result = {
            # 物体信息
            "object_found": selected_object is not None,
            "object_label": selected_object["label"] if selected_object else None,
            "object_conf": selected_object["conf"] if selected_object else None,
            "object_box": selected_object["box"] if selected_object else None,
            "object_center": smooth_center,
            "object_area": smooth_area,
            "object_size": smooth_object_size,  # 新增：线性尺寸

            # 手部信息
            "hand_found": hand_info["hand_found"],
            "hand_center": list(hand_info["center"]) if hand_info["center"] else None,
            "hand_open": hand_info["hand_open"],
            "hand_size": smooth_hand_size,  # 使用平滑后的手部尺寸

            # 帧信息
            "frame_width": w,
            "frame_height": h,
            "timestamp": time.time()
        }

        return result