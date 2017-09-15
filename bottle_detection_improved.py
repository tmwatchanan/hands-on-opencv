import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 1

while(True):
    ret, im = cap.read()
    flip_im = cv2.flip(im, 1)
    im_hsv = cv2.cvtColor(flip_im, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 0, 0], dtype=np.uint8)
    upper_red = np.array([30, 255, 255], dtype=np.uint8)

    redmask = cv2.inRange(im_hsv, lower_red, upper_red) # (H,S,V) - lower bound and upper bound respectively
    res = cv2.bitwise_and(flip_im, flip_im, mask=redmask)
    cv2.imshow('redmask', res)
    redpixel = np.sum(redmask)

    # greenmask = cv2.inRange(flip_im, (15, 55, 15), (50, 170, 50))  # (B,G,R) - lower bound and upper bound respectively
    # cv2.imshow('greenmask', greenmask)
    # greenpixel = np.sum(greenmask)
    #
    # bluemask = cv2.inRange(flip_im, (60, 15, 15), (160, 70, 30))  # (B,G,R) - lower bound and upper bound respectively
    # cv2.imshow('bluemask', bluemask)
    # bluepixel = np.sum(bluemask)

    # if (redpixel > 300000) & (redpixel > bluepixel):
    if (redpixel > 300000):
        cv2.putText(flip_im, 'Coke', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=5)
    # if (greenpixel > 200000):
    #     cv2.putText(flip_im, 'Milk', (50, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), thickness=5)
    # if (bluepixel > 200000):
    #     cv2.putText(flip_im, 'Pepsi', (50, 300), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), thickness=5)

    cv2.imshow('camera', flip_im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()