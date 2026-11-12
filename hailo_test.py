import cv2
import time
import threading
from HRT_0_Async_Inferencer import HailoDetector

latest = None

def camera(cap):
    global latest
    while True:
        ret, frame = cap.read()
        if ret:
            latest = frame

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    detector = HailoDetector("yolov8s.hef")

    threading.Thread(target=camera, args=(cap,), daemon=True).start()

    while True:
        if latest is None:
            continue

        frame = latest.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)

        start = time.perf_counter()
        detected = detector.run(frame)
        cv2.imwrite("detected_frame.jpg", detected)

        time.sleep(max(0, 1.0 - (time.perf_counter() - start)))

if __name__ == "__main__":
    main()