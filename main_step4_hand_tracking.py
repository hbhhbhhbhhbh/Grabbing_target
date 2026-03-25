import cv2
from perception.camera_stream import open_camera
from perception.hand_tracker import HandTracker


def main():
    cap = open_camera()
    tracker = HandTracker(model_path="hand_landmarker.task")

    print("Hand tracking started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        hand_info = tracker.detect(frame)
        frame = tracker.draw(frame, hand_info)

        cv2.imshow("Step 4 - Hand Tracking (Tasks API)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()