import math

class GuidanceController:
    def __init__(
        self,
        x_threshold=60,
        y_threshold=60,
        # 线性尺寸比例阈值（手/物体），应接近 1.0
        depth_ratio_low=0.75,       # 手比物体小 -> 手更远 -> 前进
        depth_ratio_high=1.25,      # 手比物体大 -> 手更近 -> 后退
        grab_distance_threshold=80,
        mirror=True,
        # 新增：握拳时忽略深度判断
        ignore_depth_when_grabbing=True
    ):
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.depth_ratio_low = depth_ratio_low
        self.depth_ratio_high = depth_ratio_high
        self.grab_distance_threshold = grab_distance_threshold
        self.mirror = mirror
        self.ignore_depth_when_grabbing = ignore_depth_when_grabbing

    def compute_distance(self, perception_data):
        """
        保留原有 2D 平面距离，供状态机和主程序继续使用
        """
        if not perception_data.get("object_found") or not perception_data.get("hand_found"):
            return None

        object_center = perception_data.get("object_center")
        hand_center = perception_data.get("hand_center")

        if object_center is None or hand_center is None:
            return None

        ox, oy = object_center
        hx, hy = hand_center

        dx = hx - ox
        dy = hy - oy

        return math.sqrt(dx * dx + dy * dy)

    def compute_offsets(self, perception_data):
        """
        返回左右、上下、2D 距离
        dx:
            dx > 0 -> 手在目标右边
            dx < 0 -> 手在目标左边
        dy:
            dy > 0 -> 手在目标上方，需要往下
            dy < 0 -> 手在目标下方，需要往上
        """
        if not perception_data.get("object_found") or not perception_data.get("hand_found"):
            return None, None, None

        object_center = perception_data.get("object_center")
        hand_center = perception_data.get("hand_center")

        if object_center is None or hand_center is None:
            return None, None, None

        ox, oy = object_center
        hx, hy = hand_center

        # mirror=True 时，按照你现有摄像头镜像显示习惯处理
        dx = (hx - ox) if self.mirror else (ox - hx)
        dy = oy - hy

        distance_2d = math.sqrt((ox - hx) ** 2 + (oy - hy) ** 2)

        return dx, dy, distance_2d

    def compute_depth_relation(self, perception_data):
        """
        用手和物体的线性尺寸比，模拟前后距离关系

        优先使用 object_size（线性），降级使用 object_area

        返回:
            depth_state:
                -1 -> 手更远，应前进
                 0 -> 深度差不多
                 1 -> 手更近，应后退
            ratio:
                hand_size / object_size (线性比)
        """
        hand_size = perception_data.get("hand_size")
        object_size = perception_data.get("object_size")
        object_area = perception_data.get("object_area")
        hand_open = perception_data.get("hand_open", True)

        if hand_size is None or hand_size <= 0:
            return None, None

        # 握拳时忽略深度判断（避免误判）
        if self.ignore_depth_when_grabbing and hand_open is False:
            return 0, None

        # 优先使用线性尺寸，降级使用面积
        if object_size is not None and object_size > 0:
            current_val = object_size
        elif object_area is not None and object_area > 0:
            current_val = math.sqrt(object_area)  # 降级：面积开根号转为线性
        else:
            return None, None

        ratio = hand_size / current_val

        if ratio < self.depth_ratio_low:
            return -1, ratio  # 手看起来比物体小 -> 手远 -> 前进
        elif ratio > self.depth_ratio_high:
            return 1, ratio   # 手看起来比物体大 -> 手近 -> 后退
        else:
            return 0, ratio   # 距离合适

    def generate_instruction(self, perception_data, state):
        """
        输出控制指令优先级：
        1. 是否找到物体/手
        2. 是否进入抓取状态
        3. 左右
        4. 上下
        5. 前后
        6. 接近抓取
        """
        if not perception_data.get("object_found"):
            return "Object not found"

        if not perception_data.get("hand_found"):
            return "Show your hand"

        dx, dy, distance_2d = self.compute_offsets(perception_data)
        depth_state, ratio = self.compute_depth_relation(perception_data)

        if dx is None or dy is None or distance_2d is None:
            return "Searching"

        # 1. 特殊状态优先
        if state == "READY_TO_GRAB":
            return "Close your hand to grab"

        if state == "GRABBED":
            return "Object grabbed"

        # 2. 左右调整
        if abs(dx) > self.x_threshold:
            return "Move right" if dx > 0 else "Move left"

        # 3. 上下调整
        if abs(dy) > self.y_threshold:
            return "Move down" if dy > 0 else "Move up"

        # 4. 前后调整（线性深度）
        # 注意：握拳时 depth_state=0，不会乱跳
        if depth_state is not None:
            if depth_state == -1:
                return "Move forward"
            elif depth_state == 1:
                return "Move backward"

        # 5. 足够接近就抓取
        if distance_2d <= self.grab_distance_threshold:
            if perception_data.get("hand_open", False):
                return "Close your hand to grab"
            return "Object grabbed"

        return "Keep moving slightly"