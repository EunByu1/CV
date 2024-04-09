import cv2 as cv

# 영상 읽기
img = cv.imread('img\soccer.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# [설정_1] Tlow=50, Thigh=150 
canny1 = cv.Canny(gray, 50, 150)

# [설정_2] Tlow=100, Thigh=200
canny2 = cv.Canny(gray, 100, 200)

cv.imshow('Original', gray)
cv.imshow('Canny1', canny1)
cv.imshow('Canny2', canny2)

cv.waitKey()
cv.destroyAllWindows()