import cv2
from perception.camera_stream import open_camera
from perception.object_detector import ObjectDetector
from perception.hand_tracker import HandTracker


def main():
    cap = open_camera()

    detector = ObjectDetector(
    model_path="yolo12n.pt",
    target_labels=["bottle", "cup"],
    conf_threshold=0.6,
    select_mode="largest"
)

    tracker = HandTracker(model_path="hand_landmarker.task")

    print("YOLO + Hand tracking started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== 目标检测 =====
        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["center"]
            label = det["label"]
            conf = det["conf"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ===== 手部追踪 =====
        hand_info = tracker.detect(frame)
        frame = tracker.draw(frame, hand_info)

        cv2.imshow("Step 5 - YOLO + Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()