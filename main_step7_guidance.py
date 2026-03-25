import cv2
from perception.camera_stream import open_camera
from perception.object_detector import ObjectDetector
from perception.hand_tracker import HandTracker
from perception.guidance_controller import GuidanceController
from perception.output_formatter import OutputFormatter


def main():
    cap = open_camera()

    detector = ObjectDetector(
        model_path="yolov8n.pt",
        target_labels=["cup"],
        conf_threshold=0.5
    )

    tracker = HandTracker(model_path="hand_landmarker.task")
    controller = GuidanceController(center_threshold=60, grab_threshold=80)
    formatter = OutputFormatter(smooth_window=5)

    frame_count = 0

    print("Stable Guidance system started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        hand_info = tracker.detect(frame)

        perception_data = formatter.format_output(
            frame=frame,
            detections=detections,
            hand_info=hand_info,
            preferred_label="cup"
        )

        instruction = controller.generate_instruction(perception_data)

        # 画所有检测框
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["center"]
            label = det["label"]
            conf = det["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 高亮选中目标 + 平滑中心点
        if perception_data["object_found"]:
            x1, y1, x2, y2 = perception_data["object_box"]
            sx, sy = perception_data["object_center"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (sx, sy), 7, (255, 0, 255), -1)

            cv2.putText(
                frame,
                f"SELECTED: {perception_data['object_label']}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

        # 画手
        frame = tracker.draw(frame, hand_info)

        # 显示指导信息
        cv2.putText(
            frame,
            f"Instruction: {instruction}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Object Found: {perception_data['object_found']}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Hand Found: {perception_data['hand_found']}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Detections: {len(detections)}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        frame_count += 1
        if frame_count % 20 == 0:
            print("\n========== STABLE DEBUG ==========")
            print("detections:", detections)
            print("selected:", perception_data)
            print("instruction:", instruction)

        cv2.imshow("Step 7 - Stable Guidance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()