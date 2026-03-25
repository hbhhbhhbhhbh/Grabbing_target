"""
使用测试视频验证系统（无需相机）

如果您没有相机权限或想在室内测试，可以使用此脚本
"""

import cv2
import numpy as np
from perception.object_detector import ObjectDetector
from perception.hand_tracker import HandTracker
from perception.guidance_controller import GuidanceController
from perception.output_formatter import OutputFormatter
from perception.tts_engine import TTSEngine
from perception.state_machine import GuidanceStateMachine
from perception.temporal_smoother import BMASmoother
from perception.roi_attention import ROIAttention
import time


def create_test_video_frame(frame_num: int) -> np.ndarray:
    """生成模拟的第一视角测试帧"""
    # 创建黑色背景（桌面）
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加一些渐变模拟桌面纹理
    for i in range(0, 480, 10):
        color = int(20 + (i % 30))
        cv2.line(frame, (0, i), (640, i), (color, color, color), 1)
    
    # 模拟杯子（绿色矩形）- 位置会轻微移动
    cup_x = 300 + int(20 * np.sin(frame_num / 30.0))
    cup_y = 200
    cup_w = 60
    cup_h = 80
    
    cv2.rectangle(frame, (cup_x, cup_y), (cup_x + cup_w, cup_y + cup_h), (0, 255, 0), -1)
    cv2.putText(frame, "CUP", (cup_x + 10, cup_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 模拟手（肤色区域）- 从底部向上移动
    hand_y = 400 - int((frame_num % 200) / 200.0 * 200)
    hand_x = 320
    
    if hand_y > 100:  # 手在画面内
        cv2.circle(frame, (hand_x, hand_y), 40, (200, 180, 150), -1)
        # 手指
        for i in range(-2, 3):
            finger_x = hand_x + i * 15
            finger_y = hand_y - 30
            cv2.circle(frame, (finger_x, finger_y), 8, (200, 180, 150), -1)
    
    return frame


def main():
    print("="*60)
    print("🎬 使用模拟视频测试优化后的系统")
    print("="*60)
    
    # 初始化所有模块
    detector = ObjectDetector(
        model_path="yolov8n.pt",
        target_labels=["cup"],
        conf_threshold=0.35,
        iou_threshold=0.45,
        input_size=640,
        label_weights={"cup": 1.2}
    )
    
    tracker = HandTracker(model_path="hand_landmarker.task")
    controller = GuidanceController(center_threshold=60, grab_threshold=80)
    formatter = OutputFormatter(smooth_window=5)
    tts = TTSEngine(repeat_interval=0.8)
    sm = GuidanceStateMachine()
    
    # 新增：时序平滑和 ROI 注意力
    box_smoother = BMASmoother(alpha=0.6, max_frames_lost=5)
    roi_attention = ROIAttention(base_radius=150, confidence_boost=0.15)
    
    # FPS 计数器
    fps_counter = 0
    fps_start_time = time.time()
    
    print("\n✅ 所有模块初始化完成")
    print("\n📊 优化配置:")
    print(f"  - 置信度阈值：0.35 (降低以减少漏检)")
    print(f"  - NMS IoU 阈值：0.45 (防止重叠框)")
    print(f"  - EMA 平滑系数：0.6")
    print(f"  - ROI 基础半径：150px")
    print(f"  - ROI 置信度提升：+0.15")
    print("\n🎬 开始播放模拟视频...")
    print("按 'q' 键退出\n")
    
    frame_num = 0
    
    while True:
        # 生成模拟帧
        frame = create_test_video_frame(frame_num)
        
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
            preferred_label="cup"
        )
        
        # ========== 4. 时序平滑 ==========
        if perception_data["object_found"]:
            temp_det = {
                "box": perception_data["object_box"],
                "center": perception_data["object_center"],
                "label": perception_data.get("selected_label", "cup"),
                "conf": perception_data.get("selected_conf", 0.5)
            }
            
            smoothed_det = box_smoother.update(temp_det)
            
            if smoothed_det:
                perception_data["object_box"] = smoothed_det["box"]
                perception_data["object_center"] = smoothed_det["center"]
        else:
            box_smoother.update(None)
        
        # ========== 5. 距离计算与状态机 ==========
        distance = controller.compute_distance(perception_data)
        if distance is None:
            distance = 9999
        
        state = sm.update_state(perception_data, distance)
        instruction = controller.generate_instruction(perception_data, state)
        
        # ========== 6. 可视化 ==========
        # 画所有检测框
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["center"]
            label = det["label"]
            conf = det["conf"]
            
            in_roi = det.get("in_roi", False)
            color = (0, 255, 255) if in_roi else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} {conf:.2f}", 
                       (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 高亮选中目标（平滑后坐标）
        if perception_data["object_found"]:
            x1, y1, x2, y2 = perception_data["object_box"]
            sx, sy = perception_data["object_center"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (sx, sy), 7, (255, 0, 255), -1)
        
        # 画手
        frame = tracker.draw(frame, hand_info)
        
        # ROI 可视化
        if hand_info.get("hand_found", False) and roi_attention.hand_center:
            cx, cy = roi_attention.hand_center
            radius = roi_attention.current_radius
            cv2.circle(frame, (cx, cy), radius, (255, 255, 0), 1)
            cv2.putText(frame, "ROI", (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 文本信息
        cv2.putText(frame, f"State: {state}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Instruction: {instruction}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, f"Object: {perception_data['object_found']}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Hand: {perception_data['hand_found']}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {int(distance)}", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        # FPS 显示
        fps_counter += 1
        if fps_counter % 10 == 0:
            current_time = time.time()
            elapsed = current_time - fps_start_time
            if elapsed > 0:
                fps = fps_counter / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}",
                           (frame.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                fps_counter = 0
                fps_start_time = current_time
        
        # 显示
        cv2.imshow("Step 9 - State Machine (Test Video)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_num += 1
        
        if frame_num % 60 == 0:
            print(f"\n📊 Frame {frame_num}:")
            print(f"  State: {state}")
            print(f"  Instruction: {instruction}")
            print(f"  Object Found: {perception_data['object_found']}")
            print(f"  Hand Found: {perception_data['hand_found']}")
            print(f"  Distance: {int(distance)}")
    
    # 清理
    cv2.destroyAllWindows()
    tts.stop()
    
    print("\n" + "="*60)
    print("✅ 测试完成!")
    print("="*60)
    print(f"\n总共处理了 {frame_num} 帧")
    print("\n💡 提示:")
    print("  - 如果看到黄色检测框，说明 ROI 注意力在工作")
    print("  - 紫色框的坐标经过 EMA 平滑，应该比较稳定")
    print("  - 如果想用真实相机，请授予终端相机权限后运行 main_step9_state_machine.py")
    print("")


if __name__ == "__main__":
    main()
