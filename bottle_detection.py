import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 1

while(True):
    ret, im = cap.read()
    flip_im = cv2.flip(im, 1)

    redmask = cv2.inRange(flip_im, (5,5,60), (40,40,180)) # (B,G,R) - lower bound and upper bound respectively
    cv2.imshow('redmask', redmask)
    redpixel = np.sum(redmask)

    greenmask = cv2.inRange(flip_im, (15, 55, 15), (50, 170, 50))  # (B,G,R) - lower bound and upper bound respectively
    cv2.imshow('greenmask', greenmask)
    greenpixel = np.sum(greenmask)

    bluemask = cv2.inRange(flip_im, (60, 15, 15), (160, 70, 30))  # (B,G,R) - lower bound and upper bound respectively
    cv2.imshow('bluemask', bluemask)
    bluepixel = np.sum(bluemask)

    if (redpixel > 300000) & (redpixel > bluepixel):
        cv2.putText(flip_im, 'Coke', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), thickness=5)
    if (greenpixel > 200000):
        cv2.putText(flip_im, 'Milk', (50, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), thickness=5)
    if (bluepixel > 200000):
        cv2.putText(flip_im, 'Pepsi', (50, 300), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), thickness=5)

    cv2.imshow('camera', flip_im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()