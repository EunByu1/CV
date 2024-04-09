import cv2
import sys

BrushSize = 5
LColor, RColor = (255,0,0), (0,0,255)

def painting(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), BrushSize, LColor, -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x,y), BrushSize, RColor, -1)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img, (x,y), BrushSize, LColor, -1)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
        cv2.circle(img, (x,y), BrushSize, RColor, -1)

    cv2.imshow('Painting', img)

img = cv2.imread('img/soccer.jpg')

if img is None:
    sys.exit('File not exists.')

cv2.namedWindow('Painting')
cv2.imshow('Painting', img)
cv2.setMouseCallback('Painting', painting)

while(True):
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break