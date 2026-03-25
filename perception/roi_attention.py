"""
ROI 注意力模块 - 基于手部位置的目标检测优化

核心功能:
1. 根据手部位置动态生成感兴趣区域 (ROI)
2. 在 ROI 内提升检测权重或进行二次筛选
3. 随状态机状态动态调整 ROI 大小
4. 支持多尺度 ROI 策略

设计原理:
- 当用户执行抓取动作时，目标物体必然在手部周围的局部区域内
- 利用手部坐标作为先验信息，可以提升该区域内的检测置信度
- 随着引导指令进行（如"靠近物体"），ROI 范围应逐渐缩小
"""

from typing import Dict, List, Optional, Tuple
import math


class ROIAttention:
    """基于手部的 ROI 注意力机制"""
    
    def __init__(
        self,
        base_radius: int = 150,
        min_radius: int = 80,
        confidence_boost: float = 0.15,
        roi_decay_rate: float = 0.95
    ):
        """
        初始化 ROI 注意力
        
        Args:
            base_radius: 基础 ROI 半径 (像素)
            min_radius: 最小 ROI 半径 (防止过度缩小)
            confidence_boost: ROI 内的置信度提升值
            roi_decay_rate: ROI 衰减率 (每帧乘以该值，直到达到 min_radius)
        """
        self.base_radius = base_radius
        self.min_radius = min_radius
        self.confidence_boost = confidence_boost
        self.roi_decay_rate = roi_decay_rate
        
        # 当前 ROI 状态
        self.current_radius = base_radius
        self.hand_center: Optional[Tuple[int, int]] = None
    
    def apply(
        self,
        detections: List[Dict],
        hand_info: Dict,
        frame_shape: Tuple[int, int, int]
    ) -> List[Dict]:
        """
        应用 ROI 注意力到检测结果
        
        Args:
            detections: 原始检测结果列表
            hand_info: 手部信息，包含"hand_found"和"center"
            frame_shape: 图像形状 (h, w, c)
        
        Returns:
            经过 ROI 注意力调整后的检测结果
        """
        if not hand_info.get("hand_found", False):
            # 没有检测到手，不应用 ROI
            self.current_radius = self.base_radius
            self.hand_center = None
            return detections
        
        h, w, _ = frame_shape
        hand_center = hand_info["center"]
        self.hand_center = hand_center
        
        # 计算 ROI 区域
        roi_mask = self._generate_roi_mask(h, w, hand_center)
        
        # 调整每个检测的置信度
        adjusted_detections = []
        for det in detections:
            adjusted_det = self._adjust_detection(det, roi_mask)
            adjusted_detections.append(adjusted_det)
        
        # 按置信度排序，优先保留 ROI 内的目标
        adjusted_detections.sort(key=lambda x: x["conf"], reverse=True)
        
        # 逐渐缩小 ROI (模拟注意力聚焦过程)
        self.current_radius = max(
            self.min_radius,
            int(self.current_radius * self.roi_decay_rate)
        )
        
        return adjusted_detections
    
    def _generate_roi_mask(
        self,
        h: int,
        w: int,
        center: Tuple[int, int]
    ) -> Dict:
        """
        生成 ROI 掩码
        
        Returns:
            包含 ROI 信息的字典：
            - cx, cy: ROI 中心
            - radius: ROI 半径
            - x1, y1, x2, y2: ROI 边界框
        """
        cx, cy = center
        radius = self.current_radius
        
        # 确保 ROI 在图像范围内
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(h, cy + radius)
        
        return {
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    
    def _adjust_detection(
        self,
        detection: Dict,
        roi_mask: Dict
    ) -> Dict:
        """
        根据 ROI 调整单个检测的置信度
        
        策略:
        1. 如果检测中心在 ROI 内，提升置信度
        2. 如果检测与 ROI 有重叠，按重叠比例提升
        3. 如果在 ROI 外且远离手部，略微降低置信度
        """
        det_center = detection.get("center", None)
        if det_center is None:
            # 计算中心
            x1, y1, x2, y2 = detection["box"]
            det_center = [(x1 + x2) // 2, (y1 + y2) // 2]
        
        dx = det_center[0] - roi_mask["cx"]
        dy = det_center[1] - roi_mask["cy"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        original_conf = detection["conf"]
        adjusted_conf = original_conf
        
        if distance <= roi_mask["radius"]:
            # 检测中心在 ROI 内，提升置信度
            adjusted_conf = min(1.0, original_conf + self.confidence_boost)
        elif distance <= roi_mask["radius"] * 1.5:
            # 在 ROI 边缘附近，适度提升
            boost_factor = 1.0 - (distance - roi_mask["radius"]) / (roi_mask["radius"] * 0.5)
            adjusted_conf = min(1.0, original_conf + self.confidence_boost * boost_factor * 0.5)
        else:
            # 在 ROI 外，略微降低置信度 (但不低于原值的 80%)
            adjusted_conf = max(original_conf * 0.95, original_conf * 0.8)
        
        # 返回调整后的结果
        result = detection.copy()
        result["conf"] = adjusted_conf
        result["in_roi"] = distance <= roi_mask["radius"]
        result["distance_to_hand"] = distance
        
        return result
    
    def reset(self):
        """重置 ROI 状态"""
        self.current_radius = self.base_radius
        self.hand_center = None
    
    def set_state_dependent_params(self, state: str):
        """
        根据状态机状态调整参数
        
        Args:
            state: 当前状态机状态字符串
        """
        if state == "SEARCHING":
            # 搜索阶段，使用较大的 ROI
            self.base_radius = 200
            self.confidence_boost = 0.1
        elif state == "APPROACHING":
            # 接近阶段，中等 ROI
            self.base_radius = 150
            self.confidence_boost = 0.15
        elif state == "GRABBING":
            # 抓取阶段，小 ROI 精确聚焦
            self.base_radius = 100
            self.confidence_boost = 0.25
        else:
            # 默认参数
            self.base_radius = 150
            self.confidence_boost = 0.15


class MultiScaleROIAttention:
    """多尺度 ROI 注意力 - 同时考虑多个半径范围"""
    
    def __init__(self, radii: List[int] = [80, 150, 250]):
        """
        Args:
            radii: 多个 ROI 半径列表，从内到外
        """
        self.radii = sorted(radii)
        self.boosts = [0.25, 0.15, 0.08]  # 对应每个半径的置信度提升
    
    def apply(
        self,
        detections: List[Dict],
        hand_info: Dict,
        frame_shape: Tuple[int, int, int]
    ) -> List[Dict]:
        """应用多尺度 ROI 注意力"""
        if not hand_info.get("hand_found", False):
            return detections
        
        h, w, _ = frame_shape
        hand_center = hand_info["center"]
        
        adjusted_detections = []
        for det in detections:
            det_center = det.get("center", None)
            if det_center is None:
                x1, y1, x2, y2 = det["box"]
                det_center = [(x1 + x2) // 2, (y1 + y2) // 2]
            
            dx = det_center[0] - hand_center[0]
            dy = det_center[1] - hand_center[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # 确定在哪一层 ROI 内
            boost = 0.0
            for i, radius in enumerate(self.radii):
                if distance <= radius:
                    boost = self.boosts[i]
                    break
            
            # 应用置信度提升
            result = det.copy()
            result["conf"] = min(1.0, det["conf"] + boost)
            result["distance_to_hand"] = distance
            adjusted_detections.append(result)
        
        adjusted_detections.sort(key=lambda x: x["conf"], reverse=True)
        return adjusted_detections


def create_roi_attention(
    mode: str = "single",
    **kwargs
) -> ROIAttention | MultiScaleROIAttention:
    """
    工厂函数：创建 ROI 注意力实例
    
    Args:
        mode: "single" 或 "multi"
        **kwargs: 传递给构造函数的参数
    
    Returns:
        ROI 注意力对象
    """
    if mode == "single":
        return ROIAttention(**kwargs)
    elif mode == "multi":
        return MultiScaleROIAttention(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
