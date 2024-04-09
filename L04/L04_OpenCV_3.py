import cv2 as cv
import sys

# 카메라와 연결 시도
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    # 비디오를 구성하는 프레임 획득
    ret, frame = cap.read()

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    cv.imshow('Video display', frame)

    # 1밀리초 동안 키보드 입력 기다림
    key = cv.waitKey(1)
    
    # 'q' 키가 들어오면 루프를 빠져나감
    if key == ord('q'):
        break

# 카메라와 연결을 끊음
cap.release()
cv.destroyAllWindows()