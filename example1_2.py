import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, im = cap.read()
    flip_im = cv2.flip(im, 1)
    cv2.imshow('camera', flip_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()