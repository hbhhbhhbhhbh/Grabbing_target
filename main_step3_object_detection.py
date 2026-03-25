import cv2
from perception.camera_stream import open_camera
from perception.object_detector import ObjectDetector

def main():
    cap = open_camera()

    detector = ObjectDetector(
        model_path="yolov12n.pt",
        target_labels=["bottle", "cup", "cell phone"],
        conf_threshold=0.5
    )

    print("Object detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

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

        cv2.imshow("Step 3 - Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()