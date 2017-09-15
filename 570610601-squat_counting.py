import os
import glob
import numpy as np
import cv2

PATH = 'SquatCounting\\'
VIDEO_EXTENSION = '*.avi'
THRESHOLD_AREA = 5000

for (file_no, filename) in enumerate(glob.glob(os.path.join(PATH, VIDEO_EXTENSION))):
    # if file_no == 0 or file_no == 1:
    #     continue
    cap = cv2.VideoCapture(filename)
    _, bg = cap.read()

    heights = [[], [], []]
    counts = [0, 0, 0]
    count_already = [False, False, False]
    x_position = [[], [], []]
    still = [False, False, False]
    while cap.isOpened():
        haveFrame, im = cap.read()
        if not haveFrame:
            break
        elif (cv2.waitKey(1) & 0xFF == ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            break

        # START PROCESSING
        diff = cv2.absdiff(im, bg)
        diff[:, 500:] = 0
        diff[:, :130] = 0
        diff[:44, :] = 0
        diff = cv2.threshold(diff, 39, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        bwmask = cv2.inRange(diff, 39, 255)
        # Median Filter
        bwmask = cv2.medianBlur(bwmask, 5)

        # Closing
        kernel = np.ones((95, 25), np.uint8)  # structuring element
        bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_CLOSE, kernel)

        # Opening
        kernel = np.ones((5, 17), np.uint8)  # structuring element
        bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_OPEN, kernel)

        # kernel = np.ones((10, 10), np.uint8)  # structuring element
        # bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_CLOSE, kernel)

        # kernel = np.ones((4, 12), np.uint8)  # structuring element
        # bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_OPEN, kernel)

        # erode_se = np.ones((7, 6), np.uint8)  # structuring element
        # bwmask = cv2.erode(bwmask, erode_se, iterations=1)
        #
        # dilate_se = np.ones((1, 8), np.uint8)  # structuring element
        # bwmask = cv2.dilate(bwmask, dilate_se, iterations=2)
        #
        # kernel = np.ones((7, 3), np.uint8)  # structuring element
        # bwmask = cv2.morphologyEx(bwmask, cv2.MORPH_OPEN, kernel)
        # #
        dilate_se = np.ones((5, 1), np.uint8)  # structuring element
        bwmask = cv2.dilate(bwmask, dilate_se, iterations=2)

        bwmask[:, 484:] = 0
        bwmask[:, :130] = 0
        bwmask[:48, :] = 0
        bwmask[63:98, 370:397] = 0
        bwmask[162:272, 396:429] = 0
        bwmask[40:128, 360:394] = 0
        bwmask[49:59, 429:460] = 0
        bwmask[133:304, 400:419] = 0

        # Contour
        contourmask = bwmask
        temp, contours, hierarchy = cv2.findContours(contourmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = contours

        selected_contours = []
        for (i, cnt) in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # area = cv2.contourArea(cnt)
            # cv2.drawContours(im, contours[i], -1, (0, 255, 0), 2)
            if (w*h > 4500) and not (w > 5*h or h > 5*w):
                # selected_contours.append(cnt)
                selected_contours.append((x, y, w, h))
                # cv2.drawContours(im, contours[i], -1, (0, 255, 0), 2)

                # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # cv2.putText(im, str(i), (x + w + 10, y + h), 0, 0.5, (255, 50, 50))
        # Skip no-contours frame
        # if len(selected_contours) < 2:
        #     # cv2.imshow('video', im)
        #     continue
        if selected_contours:
            dtype = [('x_pos', int), ('y_pos', int), ('width', int), ('height', int)]
            contours_np_array = np.array(selected_contours, dtype=dtype)
            contours_np_array = np.sort(contours_np_array, order='x_pos')
            # print(contours_np_array)

            for (i, cnt) in enumerate(contours_np_array):
                if i >= 3:
                    break
                # cv2.drawContours(im, cnt, -1, (0, 255, 0), 2)
                x, y, w, h = cnt['x_pos'], cnt['y_pos'], cnt['width'], cnt['height']
                x_position[i].append(x)
                cv2.rectangle(im, (x, y), (min(x + w, im.shape[1]), min(y + h, im.shape[0])), (0, 0, 255), 2)
                # cv2.putText(im, str(i), (min(x + w + 10, im.shape[0]), min(y + h, im.shape[1])), 0, 0.5, (255, 50, 50))

                heights[i].append(h)

                if not still[i]:
                    if len(x_position[i]) >= 15:
                        if abs(x_position[i][-1] - x_position[i][-15]) < 1:
                            heights[i][0] = h
                            # cv2.putText(im, "Still", (x + int(w/2), y + int(h/2)), 0, 0.5, (255, 50, 50))
                            still[i] = True
                        # print("still[", i, "] -> ", heights[i])
                elif len(heights[i]) >= 10:
                    # if (heights[i][-1] < heights[i][-2]) and not down_before[i]:
                    #     down_before[i] = True
                    # if (heights[i][-1] > 1.2 * heights[i][0]) and down_before[i]:
                    #     counts[i] = counts[i] + 1
                    #     down_before[i] = False
                    # print("heights[", i, "] -> ", heights[i])
                    if (heights[i][-1] < 0.765 * heights[i][0]) and not count_already[i]:
                        if counts[i] < 17:
                            counts[i] = counts[i] + 1
                        # cv2.putText(im, "DOWN", (x + int(w/2), y + int(h/2)), 0, 1.5, (55, 50, 250), thickness=2)
                        # print("down[", i, "] -> ", heights[i])
                        count_already[i] = True
                    elif (heights[i][-1] > 0.82 * heights[i][0]) and count_already[i]:
                        # cv2.putText(im, "UP", (x + int(w/2), y + int(h/2)), 0, 1.5, (55, 50, 250), thickness=2)
                        # print("up[", i, "] -> ", heights[i])
                        count_already[i] = False
                    cv2.putText(im, str(counts[i]), (x, y - 5), 0, 1.5, (255, 50, 50), thickness=7)

        # if selected_contours:
        #     sorted_contours = []
        #     index_contours = []
        #     for (i, cnt) in enumerate(selected_contours):
        #         _, y, _, _ = cv2.boundingRect(cnt)
        #         index_contours.append(y)
        #     arg_contours = sorted(range(len(index_contours)), key=lambda k: index_contours[k])
        #
        #     for (i, cnt) in enumerate(arg_contours):
        #         x, y, w, h = cv2.boundingRect(contours[cnt])
        #
        #         # high[i].append(h)
        #
        #         cv2.drawContours(im, contours[cnt], -1, (0, 255, 0), 2)
        #         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #         cv2.putText(im, str(i), (x + w + 10, y + h), 0, 0.5, (255, 50, 50))
        #
        #         # cv2.putText(im, str(count), (30, 30), 0, 0.5, (0, 250, 50))
        #         # cv2.putText(im, str(high[cnt]), (x + w + 10, y + h + 10), 0, 0.5, (0, 250, 50))
        # cv2.imshow('bwmask', bwmask)
        # Show video
        cv2.imshow('video', im)
    cap.release()
    # cv2.destroyAllWindows()

    while True:
        if (cv2.waitKey(1) & 0xFF == ord('a')):
            break
        elif (cv2.waitKey(1) & 0xFF == ord('q')):
            cap.release()
            cv2.destroyAllWindows()
            exit(1)
