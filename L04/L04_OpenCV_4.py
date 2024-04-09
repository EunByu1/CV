import cv2 as cv
import numpy as np
import sys

# 카메라와 연결 시도
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

frames = []
while True:
    # 비디오를 구성하는 프레임 획득
    ret, frame = cap.read()

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    cv.imshow('Video display', frame)

    # 1밀리초 동안 키보드 입력 기다림
    key = cv.waitKey(1)
    
    # 'c' 키가 들어오면 프레임을 리스트에 추가 
    if key == ord('c'):
        frames.append(frame)

    # 'q' 키가 들어오면 루프를 빠져나감
    if key == ord('q'):
        break

# 카메라와 연결을 끊음
cap.release()
cv.destroyAllWindows()

# 수집된 영상이 있으면
if len(frames) > 0:
    imgs = frames[0]

    # 최대 3개까지 이어 붙임
    for i in range(1, min(3, len(frames))):
        imgs = np.hstack((imgs, frames[i]))

    cv.imshow('collected images', imgs)

    cv.waitKey()
    cv.destroyAllWindows()

print(len(frames))
print(frames[0].shape)
print(type(imgs))
print(imgs.shape)