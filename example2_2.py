import numpy as np
import cv2

# cap = cv2.VideoCapture('D:\Engineering\Academic Year 2560, 1st Semester\261458 Machine Vision (3)\Hands-on-OpenCV\TestCV01.avi')
cap = cv2.VideoCapture('TestCV01.avi')

_, bg = cap.read()

while (cap.isOpened()):
    haveFrame, im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    diff = cv2.absdiff(im, bg)
    # cv2.imshow('diff video', diff)

    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    bwmask = cv2.inRange(diff, 40, 255)

    # Median Filter
    bwmask = cv2.medianBlur(bwmask, 5)

    contourmask = bwmask
    temp, contours, hierarchy = cv2.findContours(contourmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.findContours(INPUT_IMAGE, MODE,)
        #INPUT_IMAGE: passed by reference (in old version)
        #MODE: finding contours mode, in addition, you can find hierarchy
        #
    # contours = list of array of points

    cv2.drawContours(im, contours, -1, (0, 0, 255), 2)

    cv2.imshow('video', im)

cap.release()
cv2.destroyAllWindows()