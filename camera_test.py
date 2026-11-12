import cv2

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"프레임 읽기 성공! 배열 크기(해상도): {frame.shape}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)
            cv2.imwrite("python_captured.jpg", frame)
        else:
            # 처음에 워밍업 있어서 몇 번 출력됨
            print("에러: 프레임을 읽어오지 못했습니다.")

    cap.release()

if __name__ == "__main__":
    main()