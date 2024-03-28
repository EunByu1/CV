import cv2 as cv
import sys

img = cv.imread('img/soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 콜백 함수
def draw(event, x, y, flags, param):
    # 마우스 왼쪽 버튼 클릭했을 때
    if event == cv.EVENT_LBUTTONDOWN:
        cv.rectangle(img, (x,y), (x+200,y+200), (0,0,255),2)

    # 마우스 오른쪽 버튼 클릭했을 때 
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.rectangle(img, (x,y), (x+100,y+100), (255,0,0), 2)

    cv.imshow('Drawing', img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

# Drawing 윈도우에 draw 콜백 함수 지정 
cv.setMouseCallback('Drawing', draw)

# 마우스 이벤트가 언제 발생할지 모르므로 무한 반복 
while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break