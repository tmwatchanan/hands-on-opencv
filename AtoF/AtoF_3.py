import numpy as np
import cv2

count = 0
charlist = "ABCDF"

# hog = cv2.HOGDescriptor((50,50),(50,50),(50,50),(50,50),9)
# (50, 50) = input image size
# (50, 50) = block size
# (50, 50) = block sliding size
# (50, 50) = cell size
# if the input image size and cell size are the same => 1 cell
# if the cell size and block size are the same => 1 block
# This is Pyramid HOG level 0
# 9 = #bins
hog = cv2.HOGDescriptor((50,50),(20,20),(10,10),(10,10),9)
# (50, 50) = input image size
# (20, 20) = block size
# (10, 10) = block sliding size => in this case: overlap 50%
# (50, 50) = cell size

label_train = np.zeros((25,1))

for char_id in range(0,5):
    for im_id in range(1,6):
        im = cv2.imread(charlist[char_id]+"//"+str(im_id)+".bmp",0)

        im = cv2.resize(im, (50, 50))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)

        if count == 0:
            features_train = h.reshape(1,-1)
        else:
            features_train = np.concatenate((features_train,h.reshape(1,-1)),axis = 0)

        label_train[count] = char_id
        count = count+1

# knn = cv2.ml.KNearest_create()
# knn.train(features_train.astype(np.float32),cv2.ml.ROW_SAMPLE,label_train.astype(np.int32))

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR) # Linear kernel
# svm.setKernel(cv2.ml.SVM_RBF) # Radial Basis Function => too complex => overfit!
svm.train(features_train.astype(np.float32), cv2.ml.ROW_SAMPLE, label_train.astype(np.int32))

for im_id in range(1,26):
    im = cv2.imread("Unknown//" + str(im_id) + ".bmp", 0)

    im = cv2.resize(im, (50, 50))
    im = cv2.GaussianBlur(im, (3, 3), 0)
    h = hog.compute(im)
    # _,result,_,_ = knn.findNearest(h.reshape(1,-1).astype(np.float32),1)
    result = svm.predict(h.reshape(1, -1).astype(np.float32))[1]
    cv2.imshow(str(im_id)+"="+charlist[result[0][0].astype(int)],im)
    cv2.moveWindow(str(im_id)+"="+charlist[result[0][0].astype(int)],100+((im_id-1)%5)*70,np.floor((im_id-1)/5).astype(int)*150)

cv2.waitKey(0)
cv2.destroyAllWindows()

