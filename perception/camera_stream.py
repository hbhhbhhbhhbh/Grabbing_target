import cv2

def open_camera(camera_id=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_id)

    # 设置分辨率（避免太高导致卡顿）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    return cap