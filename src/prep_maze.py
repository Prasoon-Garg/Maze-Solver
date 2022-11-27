import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(org_img):
    img = org_img.copy()
    # img = cv2.copyMakeBorder(img,3,3,3,3,cv2.BORDER_CONSTANT, None, value = (255,255,255))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # plt.imshow(img,cmap='gray')
    # plt.show()
    return img

# org_img = cv2.imread('/home/prash/Documents/DIP_Project_new/data/inputs/medium.png')
# img=preprocess(org_img)

# plt.imshow(img,cmap = 'gray')
# plt.show()