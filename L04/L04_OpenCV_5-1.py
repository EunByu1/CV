import cv2 as cv
import sys

img = cv.imread('img/soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 직사각형 그리기 
cv.rectangle(img, (290,780), (620,950), (0,0,255),2)

# 글씨 쓰기
cv.putText(img, 'mouse', (290,770), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

cv.imshow('Draw', img)

cv.waitKey()
cv.destroyAllWindows()