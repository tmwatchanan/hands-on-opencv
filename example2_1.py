import numpy as np
import cv2

# cap = cv2.VideoCapture('D:\Engineering\Academic Year 2560, 1st Semester\261458 Machine Vision (3)\Hands-on-OpenCV\TestCV01.avi')
cap = cv2.VideoCapture('TestCV01.avi')

while (cap.isOpened()):
    haveFrame, im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    cv2.imshow('video', im)

cap.release()
cv2.destroyAllWindows()