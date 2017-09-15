import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 1

while(True):
    ret, im = cap.read()

    if count > 5:
        im0 = im1
        im1 = im2
        im2 = im3
        im3 = flip_im
        flip_im = cv2.flip(im, 1)
        out = (0.2*im0 + 0.2*im1 + 0.2*im2 + 0.2*im3 + 0.2*flip_im).astype(np.uint8) # normalize for preventing data overflow from data type
        cv2.imshow('camera', out)
    else:
        im0 = im1 = im2 = im3 = flip_im = cv2.flip(im, 1)
        cv2.imshow('camera', flip_im)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()