import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 1

while(True):
    ret, im = cap.read()
    flip_im = cv2.flip(im, 1)

    mask = cv2.inRange(flip_im, (0,0,65), (50,50,255)) # (B,G,R) - lower bound and upper bound respectively
    cv2.imshow('mask', mask)

    if (np.sum(mask > 60000)):
        cv2.putText(flip_im, 'Coke', (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255))

    cv2.imshow('camera', flip_im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()