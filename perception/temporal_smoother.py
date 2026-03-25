"""
时序平滑模块 - 用于视频流中检测框的稳定性优化

核心功能:
1. EMA(指数移动平均) 平滑边界框坐标
2. 处理检测丢失时的外推预测
3. 支持多目标跟踪平滑

数学原理:
P_t = α * P_raw + (1-α) * P_{t-1}
其中:
- P_raw: 当前帧模型输出的原始坐标
- P_{t-1}: 上一帧的平滑坐标
- α: 平滑系数 (默认 0.6)
"""

import numpy as np
from typing import Optional, Dict, List


class BMASmoother:
    """边界框 EMA 平滑器"""
    
    def __init__(self, alpha: float = 0.6, max_frames_lost: int = 5):
        """
        初始化平滑器
        
        Args:
            alpha: 平滑系数，范围 (0, 1)。越大越跟随当前帧，越小越平滑
            max_frames_lost: 最大可容忍丢失帧数，超过后重置状态
        """
        self.alpha = alpha
        self.max_frames_lost = max_frames_lost
        
        # 状态变量
        self.prev_box: Optional[List[int]] = None  # [x1, y1, x2, y2]
        self.prev_center: Optional[List[int]] = None  # [cx, cy]
        self.frames_lost = 0
        
        # 速度估计 (用于外推)
        self.velocity: Optional[np.ndarray] = None
    
    def update(self, detection: Optional[Dict]) -> Optional[Dict]:
        """
        更新平滑状态并返回平滑后的检测结果
        
        Args:
            detection: 当前帧的检测结果，包含"box"和"center"字段
                      如果为 None 或检测丢失，则进行外推预测
        
        Returns:
            平滑后的检测结果，格式与输入相同
        """
        if detection is not None and "box" in detection:
            # 有检测结果，应用 EMA 平滑
            current_box = detection["box"]
            current_center = detection.get("center", None)
            
            if self.prev_box is None:
                # 第一帧，直接返回原值
                self.prev_box = current_box.copy()
                if current_center:
                    self.prev_center = current_center.copy()
                self.frames_lost = 0
                return detection
            
            # 对边界框应用 EMA
            smoothed_box = self._ema_smooth(current_box, self.prev_box)
            
            # 计算速度 (用于后续外推)
            if self.velocity is None:
                self.velocity = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                box_diff = np.array(current_box) - np.array(self.prev_box)
                self.velocity = 0.5 * self.velocity + 0.5 * box_diff
            
            # 如果有 center，也对其进行平滑
            if current_center and self.prev_center is not None:
                smoothed_center = self._ema_smooth(current_center, self.prev_center)
            elif current_center:
                smoothed_center = current_center
            else:
                # 从平滑后的 box 推算 center
                smoothed_center = [
                    (smoothed_box[0] + smoothed_box[2]) // 2,
                    (smoothed_box[1] + smoothed_box[3]) // 2
                ]
            
            # 更新状态
            self.prev_box = smoothed_box
            self.prev_center = smoothed_center
            self.frames_lost = 0
            
            # 返回平滑后的结果
            result = detection.copy()
            result["box"] = smoothed_box
            result["center"] = smoothed_center
            return result
        
        else:
            # 检测丢失，进行外推预测
            self.frames_lost += 1
            
            if self.frames_lost > self.max_frames_lost or self.prev_box is None:
                # 超过最大丢失帧数，重置状态
                self.reset()
                return None
            
            # 基于速度外推
            predicted_box = self._predict_next()
            predicted_center = self._predict_center()
            
            # 更新状态
            self.prev_box = predicted_box
            self.prev_center = predicted_center
            
            # 返回预测结果
            return {
                "box": predicted_box,
                "center": predicted_center,
                "label": "predicted",
                "conf": 0.0,
                "area": max(0, predicted_box[2] - predicted_box[0]) * 
                        max(0, predicted_box[3] - predicted_box[1])
            }
    
    def _ema_smooth(self, current: List[int], previous: List[int]) -> List[int]:
        """
        应用指数移动平均平滑
        
        公式：P_t = α * P_raw + (1-α) * P_{t-1}
        """
        smoothed = []
        for curr_val, prev_val in zip(current, previous):
            smooth_val = int(self.alpha * curr_val + (1 - self.alpha) * prev_val)
            smoothed.append(smooth_val)
        return smoothed
    
    def _predict_next(self) -> List[int]:
        """基于速度外推下一帧的边界框"""
        if self.prev_box is None or self.velocity is None:
            return self.prev_box
        
        predicted = np.array(self.prev_box) + self.velocity
        return [int(x) for x in predicted]
    
    def _predict_center(self) -> List[int]:
        """基于速度外推下一帧的中心点"""
        if self.prev_center is None or self.velocity is None:
            return self.prev_center
        
        # 中心点速度是边界框速度的一半
        center_velocity = self.velocity[:2] / 2.0
        predicted = np.array(self.prev_center) + center_velocity
        return [int(x) for x in predicted]
    
    def reset(self):
        """重置平滑器状态"""
        self.prev_box = None
        self.prev_center = None
        self.frames_lost = 0
        self.velocity = None
    
    def get_state(self) -> Dict:
        """获取当前状态信息"""
        return {
            "prev_box": self.prev_box,
            "prev_center": self.prev_center,
            "frames_lost": self.frames_lost,
            "velocity": self.velocity.tolist() if self.velocity is not None else None
        }


class MultiObjectSmoother:
    """多目标跟踪平滑器 - 为每个目标维护独立的 EMA 状态"""
    
    def __init__(self, alpha: float = 0.6, max_age: int = 30):
        """
        Args:
            alpha: 平滑系数
            max_age: 跟踪器最大存活帧数 (帧)
        """
        self.alpha = alpha
        self.max_age = max_age
        
        # 跟踪器字典：track_id -> smoother
        self.trackers: Dict[int, BMASmoother] = {}
        self.next_track_id = 0
        self.age: Dict[int, int] = {}  # 每个跟踪器的年龄
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新所有目标的平滑状态
        
        Args:
            detections: 当前帧的所有检测结果列表
        
        Returns:
            平滑后的检测结果列表，每个结果增加"track_id"字段
        """
        if not detections:
            # 没有检测结果，但仍然更新现有跟踪器
            self._update_no_detection()
            return []
        
        # 简单的关联策略：基于 IoU 匹配
        # TODO: 可以使用匈牙利算法优化
        
        smoothed_detections = []
        used_trackers = set()
        
        for det in detections:
            best_tracker_id = self._find_best_tracker(det)
            
            if best_tracker_id is not None:
                # 使用现有跟踪器
                tracker = self.trackers[best_tracker_id]
                smoothed_det = tracker.update(det)
                if smoothed_det:
                    smoothed_det["track_id"] = best_tracker_id
                    smoothed_detections.append(smoothed_det)
                    used_trackers.add(best_tracker_id)
                    self.age[best_tracker_id] = 0
            else:
                # 创建新跟踪器
                track_id = self.next_track_id
                self.next_track_id += 1
                
                new_smoother = BMASmoother(alpha=self.alpha)
                smoothed_det = new_smoother.update(det)
                if smoothed_det:
                    smoothed_det["track_id"] = track_id
                    smoothed_detections.append(smoothed_det)
                    
                    self.trackers[track_id] = new_smoother
                    self.age[track_id] = 0
                    used_trackers.add(track_id)
        
        # 更新未使用的跟踪器
        for track_id in list(self.trackers.keys()):
            if track_id not in used_trackers:
                self.age[track_id] += 1
                if self.age[track_id] > self.max_age:
                    # 删除超龄跟踪器
                    del self.trackers[track_id]
                    del self.age[track_id]
        
        return smoothed_detections
    
    def _find_best_tracker(self, detection: Dict) -> Optional[int]:
        """基于 IoU 为当前检测找到最佳匹配的跟踪器"""
        if not self.trackers:
            return None
        
        det_box = detection["box"]
        best_iou = 0.0
        best_tracker_id = None
        
        for track_id, tracker in self.trackers.items():
            state = tracker.get_state()
            if state["prev_box"] is None:
                continue
            
            iou = self._compute_iou(det_box, state["prev_box"])
            if iou > best_iou:
                best_iou = iou
                best_tracker_id = track_id
        
        # IoU 阈值，低于此值认为不匹配
        return best_tracker_id if best_iou > 0.1 else None
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算两个边界框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _update_no_detection(self):
        """在没有检测结果时更新所有跟踪器"""
        for track_id in list(self.trackers.keys()):
            tracker = self.trackers[track_id]
            tracker.update(None)
            self.age[track_id] += 1
            
            if self.age[track_id] > self.max_age:
                del self.trackers[track_id]
                del self.age[track_id]
    
    def reset(self):
        """重置所有跟踪器"""
        self.trackers.clear()
        self.age.clear()
        self.next_track_id = 0
