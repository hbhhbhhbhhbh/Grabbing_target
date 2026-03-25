import cv2
from perception.camera_stream import open_camera
from perception.object_detector import ObjectDetector
from perception.hand_tracker import HandTracker
from perception.output_formatter import format_output


def main():
    cap = open_camera()

    detector = ObjectDetector(
        model_path="yolov12n.pt",
        target_labels=["bottle", "cup", "cell phone"],
        conf_threshold=0.5
    )

    tracker = HandTracker(model_path="hand_landmarker.task")

    target_object_name = "bottle"

    print("Step 6 started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        hand_info = tracker.detect(frame)

        # ===== 格式化输出 =====
        result = format_output(
            frame=frame,
            detections=detections,
            hand_info=hand_info,
            preferred_label=target_object_name
        )

        # ===== 画目标框 =====
        if result["object_found"]:
            x1, y1, x2, y2 = result["object_box"]
            cx, cy = result["object_center"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f'{result["object_label"]} {result["object_conf"]:.2f}',
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ===== 画手 =====
        frame = tracker.draw(frame, hand_info)

        # ===== 左上角显示核心数据 =====
        cv2.putText(frame, f'Target: {target_object_name}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f'Object Found: {result["object_found"]}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f'Hand Found: {result["hand_found"]}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ===== 在终端打印结果（先别每帧都打印太多，可以每隔几帧看一次）=====
        print(result)

        cv2.imshow("Step 6 - Unified Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()