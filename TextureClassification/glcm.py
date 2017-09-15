import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops


count = 0
class_list = ["Beef", "Omelet", "Spaghetti"]
L = 10
label_train = np.zeros((15*3,1))
features_train = np.zeros((15*3,4 * L * 3))
for class_id in range(0,3):
    for im_id in range(1,16):
        im = cv2.imread(class_list[class_id]+"//"+str(im_id)+".jpg")
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.resize(im_gray, (50, 50))
        # By default, gray image will have 256 x 256 dimensions
        # We will divide it by 16, that is normalizing all pixels into range 0-15
        # So, it will have 16 x 16 instead
        im_gray = (im_gray / 16).astype(np.uint8)
        # We do not use glcm directly, but the statistical numbers instead
        # range(1, L + 1): 10 offsets
        # [0, np.pi / 4, np.pi / 2]: 3 directions, that are, 0, 90, and 180
        glcm = greycomatrix(im_gray, range(1, L + 1), [0, np.pi / 4, np.pi / 2], 16, symmetric=True, normed=True)
        glcm_props = np.zeros(4 * L * 3) # 4 properties x 3 directions x 10 offsets = 120 features
        glcm_props[0:(L * 3)] = greycoprops(glcm, 'ASM').reshape(1, -1)[0] # Uniformity
        glcm_props[(L * 3):(L * 3 * 2)] = greycoprops(glcm, 'contrast').reshape(1, -1)[0]
        glcm_props[(L * 3 * 2):(L * 3 * 3)] = greycoprops(glcm, 'homogeneity').reshape(1, -1)[0]
        glcm_props[(L * 3 * 3):(L * 3 * 4)] = greycoprops(glcm, 'correlation').reshape(1, -1)[0]
        features_train[count] = glcm_props # 120 features x 45 images --> 45 rows and each row has 120 features
        label_train[count] = class_id
        count = count+1

svm = cv2.ml.SVM_create() # Train using SVM
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features_train.astype(np.float32), cv2.ml.ROW_SAMPLE,label_train.astype(np.int32))

# Loop inside Unknown folder
for im_id in range(1,16):
    im = cv2.imread("Unknown//" + str(im_id) + ".jpg")
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.resize(im_gray, (50, 50))
    im_gray = (im_gray / 16).astype(np.uint8)
    glcm = greycomatrix(im_gray, range(1, L + 1), [0, np.pi / 4, np.pi / 2], 16, symmetric=True, normed=True)
    glcm_props = np.zeros(4 * L * 3)
    glcm_props[0:(L * 3)] = greycoprops(glcm, 'ASM').reshape(1, -1)[0]
    glcm_props[(L * 3):(L * 3 * 2)] = greycoprops(glcm, 'contrast').reshape(1, -1)[0]
    glcm_props[(L * 3 * 2):(L * 3 * 3)] = greycoprops(glcm, 'homogeneity').reshape(1, -1)[0]
    glcm_props[(L * 3 * 3):(L * 3 * 4)] = greycoprops(glcm, 'correlation').reshape(1, -1)[0]
    # Predict the result
    result = svm.predict(glcm_props.reshape(1,-1).astype(np.float32))[1]
    cv2.imshow(str(im_id)+"="+class_list[result[0][0].astype(int)],im)
    cv2.moveWindow(str(im_id)+"="+class_list[result[0][0].astype(int)],100+((im_id-1)%5)*200,np.floor((im_id-1)/5).astype(int)*200)

cv2.waitKey(0)
cv2.destroyAllWindows()
