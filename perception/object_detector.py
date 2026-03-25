from ultralytics import YOLO
import cv2
import math

class ObjectDetector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        target_labels=None,
        conf_threshold=0.35,      # 降低以减少漏检
        iou_threshold=0.45,       # NMS IoU 阈值
        input_size=640,          # 输入分辨率
        select_mode="largest",   # "largest" 或 "highest_conf"
        label_weights=None,      # 可选的类别权重
    ):
        self.model = YOLO(model_path)
        self.target_labels = target_labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.select_mode = select_mode
        self.label_weights = label_weights or {}

    def detect(self, frame):
        """
        输入：BGR frame (OpenCV 原图)
        输出：detections 列表，包含修正后的原图坐标和线性尺寸
        """
        original_shape = frame.shape[:2]  # (H, W)
        h_orig, w_orig = original_shape
        
        # 1. 预处理：Resize 用于推理
        detect_frame = frame
        scale_x, scale_y = 1.0, 1.0
        
        if self.input_size and frame.shape[0] != self.input_size:
            detect_frame = cv2.resize(frame, (self.input_size, self.input_size))
            # 计算缩放比例，用于还原坐标
            scale_x = w_orig / self.input_size
            scale_y = h_orig / self.input_size
        
        # 2. 推理
        results = self.model(
            detect_frame, 
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )
        
        detections = []

        for result in results:
            boxes = result.boxes
            names = result.names

            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # 应用类别权重
                effective_conf = conf
                label_name = names[cls_id]
                if label_name in self.label_weights:
                    effective_conf = conf * self.label_weights[label_name]
                
                if effective_conf < self.conf_threshold:
                    continue

                # 3. 坐标还原 (关键修复)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 将 640x640 的坐标还原回原图坐标
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # 防止越界
                x1 = max(0, min(x1, w_orig))
                y1 = max(0, min(y1, h_orig))
                x2 = max(0, min(x2, w_orig))
                y2 = max(0, min(y2, h_orig))

                label = label_name

                if self.target_labels is not None and label not in self.target_labels:
                    continue

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # 计算尺寸
                box_w = x2 - x1
                box_h = y2 - y1
                area = max(0, box_w) * max(0, box_h)
                
                # 【新增】线性尺寸：用于配合 hand_tracker 的关键点距离
                # 使用宽高的几何平均值，更接近视觉上的"直径"
                object_size = math.sqrt(area) 

                detections.append({
                    "label": label,
                    "conf": effective_conf,
                    "box": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "area": area,          # 保留兼容
                    "object_size": object_size, # 新增：线性尺寸
                    "class_id": cls_id
                })

        return detections

    def select_best_detection(self, detections, preferred_label=None):
        """
        从检测结果里选一个最适合给上层使用的目标
        """
        candidates = detections

        if preferred_label is not None:
            filtered = [d for d in detections if d["label"] == preferred_label]
            if filtered:
                candidates = filtered

        if not candidates:
            return None

        if self.select_mode == "highest_conf":
            return max(candidates, key=lambda d: d["conf"])

        # 默认选面积最大的，通常更接近用户，更适合抓取演示
        return max(candidates, key=lambda d: d["area"])

    def draw(self, frame, detection):
        """
        辅助绘图函数
        """
        if not detection:
            return frame
            
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        label = f"{detection['label']} {detection['conf']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 显示物体尺寸 (调试深度用)
        if "object_size" in detection:
            cv2.putText(
                frame, 
                f"Obj Size: {detection['object_size']:.1f}", 
                (x1, y2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
            
        return frame