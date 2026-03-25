import cv2
import time
import math
from perception.camera_stream import open_camera
from perception.object_detector import ObjectDetector
from perception.hand_tracker import HandTracker
from perception.guidance_controller import GuidanceController
from perception.output_formatter import OutputFormatter
from perception.tts_engine import TTSEngine
from perception.state_machine import GuidanceStateMachine
from perception.temporal_smoother import BMASmoother
from perception.roi_attention import ROIAttention


def compute_box_area(box):
    """计算包围盒面积"""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    return area if area > 0 else None


def compute_object_size(box):
    """计算物体线性尺寸（用于深度判断）"""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    return math.sqrt(area) if area > 0 else None


def main():
    cap = open_camera()
    
    detector = ObjectDetector(
        model_path="best.pt",
        target_labels=["pocari"],
        conf_threshold=0.35,
        iou_threshold=0.45,
        input_size=480,
        label_weights={"pocari": 1.2}
    )

    tracker = HandTracker(model_path="hand_landmarker.task")

    # 修改：深度比例阈值调整为线性比（手尺寸/物体尺寸）
    controller = GuidanceController(
        x_threshold=120,
        y_threshold=200,
        depth_ratio_low=0.75,      # 线性比阈值（原 0.85）
        depth_ratio_high=1.25,     # 线性比阈值（原 1.15）
        grab_distance_threshold=80,
        mirror=True,
        ignore_depth_when_grabbing=True  # 新增：握拳时忽略深度判断
    )

    formatter = OutputFormatter(smooth_window=5)
    tts = TTSEngine(repeat_interval=2.0, min_change_interval=0.7, poll_interval=0.1)
    sm = GuidanceStateMachine()

    # 时序平滑与 ROI 注意力
    box_smoother = BMASmoother(alpha=0.6, max_frames_lost=5)
    roi_attention = ROIAttention(
        base_radius=150,
        confidence_boost=0.15
    )

    # FPS 监控
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    frame_count = 0
    instruction_candidate = None
    instruction_candidate_count = 0
    instruction_stable_frames = 5

    print("State-machine guidance system started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ========== 1. 原始检测 ==========
        detections = detector.detect(frame)
        hand_info = tracker.detect(frame)

        # ========== 2. ROI 注意力优化 ==========
        if hand_info.get("hand_found", False):
            detections = roi_attention.apply(detections, hand_info, frame.shape)

        # ========== 3. 格式化输出 ==========
        perception_data = formatter.format_output(
            frame=frame,
            detections=detections,
            hand_info=hand_info,
            preferred_label="pocari"
        )

        # ========== 4. 时序平滑（对选中目标） ==========
        if perception_data["object_found"]:
            temp_det = {
                "box": perception_data["object_box"],
                "center": perception_data["object_center"],
                "label": perception_data.get("object_label", "cup"),
                "conf": perception_data.get("object_conf", 0.5)
            }

            smoothed_det = box_smoother.update(temp_det)

            if smoothed_det:
                perception_data["object_box"] = smoothed_det["box"]
                perception_data["object_center"] = smoothed_det["center"]

                # 关键：平滑后同步更新 object_area 和 object_size
                smoothed_area = compute_box_area(smoothed_det["box"])
                perception_data["object_area"] = smoothed_area
                
                # 新增：计算线性尺寸供深度判断
                smoothed_size = compute_object_size(smoothed_det["box"])
                perception_data["object_size"] = smoothed_size
        else:
            box_smoother.update(None)

        # 双保险：如果 formatter 没给出面积/尺寸，这里兜底补一个
        if perception_data.get("object_area") is None and perception_data.get("object_box") is not None:
            perception_data["object_area"] = compute_box_area(perception_data["object_box"])
        
        if perception_data.get("object_size") is None and perception_data.get("object_box") is not None:
            perception_data["object_size"] = compute_object_size(perception_data["object_box"])

        # ========== 5. 距离计算与状态机 ==========
        distance = controller.compute_distance(perception_data)
        if distance is None:
            distance = 9999

        state = sm.update_state(perception_data, distance)
        instruction = controller.generate_instruction(perception_data, state)

        # ========== 6. 语音播报 ==========
        if instruction in [
            "Move left",
            "Move right",
            "Move up",
            "Move down",
            "Move forward",
            "Move backward",
            "Close your hand to grab",
            "Searching for the cup",
            "Keep moving slightly",
            "Object grabbed",
            "Show your hand",
            "Object not found"
        ]:
            if instruction == instruction_candidate:
                instruction_candidate_count += 1
            else:
                instruction_candidate = instruction
                instruction_candidate_count = 1

            if instruction_candidate_count >= instruction_stable_frames:
                tts.speak(instruction)
        else:
            instruction_candidate = None
            instruction_candidate_count = 0
            tts.clear()

        # ========== 7. 绘制所有检测框 ==========
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["center"]
            label = det["label"]
            conf = det["conf"]

            in_roi = det.get("in_roi", False)
            color = (0, 255, 255) if in_roi else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # ========== 8. 高亮选中目标（平滑后） ========== 
        if perception_data["object_found"] and perception_data["object_box"] is not None:
            x1, y1, x2, y2 = perception_data["object_box"]
            sx, sy = perception_data["object_center"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (sx, sy), 7, (255, 0, 255), -1)

        # ========== 9. 绘制手 ==========
        frame = tracker.draw(frame, hand_info)

        # ========== 10. 绘制 ROI ==========
        if hand_info.get("hand_found", False) and roi_attention.hand_center:
            cx, cy = roi_attention.hand_center
            radius = roi_attention.current_radius
            cv2.circle(frame, (cx, cy), radius, (255, 255, 0), 1)
            cv2.putText(
                frame,
                "ROI",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

        # ========== 11. 计算深度比例（线性比） ==========
        ratio = None
        hand_size = perception_data.get("hand_size")
        object_size = perception_data.get("object_size")
        object_area = perception_data.get("object_area")

        if hand_size is not None and object_size is not None and object_size > 0:
            ratio = hand_size / object_size  # 线性/线性
        elif hand_size is not None and object_area is not None and object_area > 0:
            ratio = hand_size / math.sqrt(object_area)  # 降级：面积开根号

        # ========== 12. 文本信息 ==========
        cv2.putText(
            frame, f"State: {state}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        cv2.putText(
            frame, f"Instruction: {instruction}", (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
        )

        cv2.putText(
            frame, f"Object Found: {perception_data['object_found']}", (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"Hand Found: {perception_data['hand_found']}", (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"Distance: {int(distance)}", (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"Hand Size: {int(hand_size) if hand_size else 0}", (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        # 新增：显示物体线性尺寸
        cv2.putText(
            frame, f"Obj Size: {int(object_size) if object_size else 0}", (10, 230),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        cv2.putText(
            frame, f"Obj Area: {int(object_area) if object_area else 0}", (10, 260),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        if ratio is not None:
            # 根据比例值变色（绿色=合适，红色=偏差大）
            ratio_color = (0, 255, 0) if 0.75 <= ratio <= 1.25 else (0, 0, 255)
            cv2.putText(
                frame, f"Depth Ratio: {ratio:.3f}", (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ratio_color, 2
            )

        cv2.putText(
            frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # ========== 13. FPS 统计 ==========
        fps_counter += 1
        if fps_counter % 10 == 0:
            current_time = time.time()
            elapsed = current_time - fps_start_time
            if elapsed > 0:
                current_fps = fps_counter / elapsed
                if current_fps < 25:
                    print(f"Warning: Low FPS detected ({current_fps:.1f}). Consider reducing input_size to 480.")
                fps_counter = 0
                fps_start_time = current_time

        # ========== 14. 调试输出 ==========
        frame_count += 1
        if frame_count % 100 == 0:
            print("\n========== STATE MACHINE DEBUG ==========")
            print("state:", state)
            print("object_found:", perception_data.get("object_found"))
            print("object_center:", perception_data.get("object_center"))
            print("object_area:", perception_data.get("object_area"))
            print("object_size:", perception_data.get("object_size"))
            print("hand_found:", perception_data.get("hand_found"))
            print("hand_center:", perception_data.get("hand_center"))
            print("hand_size:", perception_data.get("hand_size"))
            print("hand_open:", perception_data.get("hand_open"))
            print("distance:", distance)
            print("depth_ratio:", ratio)
            print("instruction:", instruction)

        cv2.imshow("Step 9 - State Machine Guidance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tts.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()