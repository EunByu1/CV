import cv2 as cv
import sys

img = cv.imread('img/soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

def draw(event, x, y, flags, param):
    global ix, iy
    
    # 마우스 왼쪽 버튼 클릭했을 때 초기 위치 저장 
    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    
    # 마우스 왼쪽 버튼 클릭했을 때 직사각형 그리기 
    elif event == cv.EVENT_LBUTTONUP:
        cv.rectangle(img, (ix,iy), (x,y), (0,0,255), 2)

    cv.imshow('Drawing', img)

cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break