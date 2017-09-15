import numpy as np
import cv2

IMAGE_RELATIVE_PATH = 'coincount2017\\'
IMAGE_FILENAME = 'coin0'
coin = []
coin_hsv = []

for i in range(1, 6):
    print('[READ]: ' + IMAGE_RELATIVE_PATH + IMAGE_FILENAME + str(i) + '.jpg')
    coin.append(cv2.imread(IMAGE_RELATIVE_PATH + IMAGE_FILENAME + str(i) + '.jpg'))
    coin_hsv.append(cv2.cvtColor(coin[i - 1], cv2.COLOR_BGR2HSV))

def extract_mask(img, color):
    # cv2.imshow('coin_hsv[idx] original', coin_hsv[idx])
    h, s, v = cv2.split(img)
    # s.fill(0)
    # v.fill(200)
    img = cv2.merge([h, s, v])
    # bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    if color is 'blue':
        LOWER = np.array([90, 110, 0], dtype=np.uint8)
        UPPER = np.array([115, 255, 255], dtype=np.uint8)
    elif color is 'yellow':
        LOWER = np.array([20, 110, 0], dtype=np.uint8)
        UPPER = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, LOWER, UPPER)
    # cv2.imshow(color + 'mask', mask)
    return mask

def preprocess_coin(bgr_img, mask):

    res = cv2.bitwise_and(bgr_img, bgr_img, mask=mask)
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('res', res)

    # opening_se = np.ones((6, 1), np.uint8) # structuring element
    opening_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25, 25)) #25,25
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_se)
    # cv2.imshow('opening 1', opening)

    closing_se = np.ones((20, 20), np.uint8) # structuring element
    # closing_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35, 35))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_se)
    # cv2.imshow('closing', closing)

    # Erode
    erode_se = np.ones((7, 7), np.uint8) # structuring element
    erode = cv2.erode(closing,erode_se,iterations=2)
    # cv2.imshow('erode', erode)

    # opening_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
    # # opening_se = np.ones((5, 5), np.uint8) # structuring element
    # opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, opening_se)
    # # cv2.imshow('opening after closing', opening)

    opening_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35, 35)) #35,35
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, opening_se)
    # cv2.imshow('opening last', opening)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.30*dist_transform.max(),255,0)
    # cv2.imshow('dist_transform', sure_fg)

    sure_fg = np.uint8(sure_fg)
    return sure_fg

for idx in range(5):
    for mask_color in ['blue', 'yellow']:
        processed_coin = preprocess_coin(coin[idx], extract_mask(coin_hsv[idx], mask_color))
        contourmask = processed_coin
        temp, contours, hierarchy = cv2.findContours(contourmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(coin[idx], contours, -1, (0, 255, 0), 2)

        # Draw bounding boxes
        count = 1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(coin[idx], (x, y), (x + w, y + h), (0, 0, 255), 2)
            if mask_color is 'blue':
                TEXT_COLOR = (50, 50, 150)
            elif mask_color is 'yellow':
                TEXT_COLOR = (150, 50, 50)
            cv2.putText(coin[idx], str(count), (x + int(w/2), y + int(h/2)), cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, thickness=3)
            count += 1

    cv2.imshow('coin[' + str(idx) + ']', coin[idx])


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
# cv2.waitKey()