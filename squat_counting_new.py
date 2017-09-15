import os
import glob
import numpy as np
import cv2

PATH = 'SquatCounting\\'
VIDEO_EXTENSION = '*.avi'
THRESHOLD_AREA = 10000

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

for filename in glob.glob(os.path.join(PATH, VIDEO_EXTENSION)):
    cap = cv2.VideoCapture(filename)
    _, bg = cap.read()
    while cap.isOpened():
        haveFrame, im = cap.read()
        if (not haveFrame):
            break
        elif (cv2.waitKey(1) & 0xFF == ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            break

        # START PROCESSING
        fgmask = fgbg.apply(im)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Show video
        cv2.imshow('video', im)
        cv2.imshow('frame', fgmask)

    cap.release()
    # cv2.destroyAllWindows()
    while True:
        if (cv2.waitKey(1) & 0xFF == ord('a')):
            break
        elif (cv2.waitKey(1) & 0xFF == ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            exit(1)
