import cv2
from perception.camera_stream import open_camera

def main():
    cap = open_camera()

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Camera Test", frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()