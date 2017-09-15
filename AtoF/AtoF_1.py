import numpy as np
import cv2

count = 0
charlist = "ABCDF"

# HOG extractor object
hog = cv2.HOGDescriptor((50,50),(50,50),(50,50),(50,50),9)
# (50, 50) = input image size
# (50, 50) = cell size
# if the input image size and cell size are the same => 1 cell
# (50, 50) = block size
# if the cell size and block size are the same => 1 block
# This is Pyramid HOG level 0
# 9 = #bins

label_train = np.zeros((25,1))

for char_id in range(0,5):
    for im_id in range(1,6):
        im = cv2.imread(charlist[char_id]+"//"+str(im_id)+".bmp",0)

        im = cv2.resize(im, (50, 50))
        # Because we use binary image, not the grey-level image
        im = cv2.GaussianBlur(im, (3, 3), 0) # smooth image for finding the actual angles, instead of only misleading 4 angles (45, 135, ...)
        h = hog.compute(im)

        if count == 0:
            features_train = h.reshape(1,-1)
        else:
            features_train = np.concatenate((features_train,h.reshape(1,-1)),axis = 0)

        label_train[count] = char_id
        count = count+1

print(features_train)
print(label_train)
cv2.waitKey(0)
cv2.destroyAllWindows()

