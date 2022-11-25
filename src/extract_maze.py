import numpy as np
import cv2
from matplotlib import pyplot as plt


def top_left(img):
    left_top = []
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] == 255:
                left_top.append(i)
                left_top.append(j)
                return left_top

def bottom_right(img):
    right_bottom = []
    h,w = img.shape
    for i in range(h-1,0,-1):
        for j in range(w-1,0,-1):
            if img[i][j] == 255:
                right_bottom.append(i)
                right_bottom.append(j)
                return right_bottom

def top_right(img):
    right_top = []
    h,w = img.shape
    for i in range(h):
        for j in range(w-1,0,-1):
            if img[i][j] == 255:
                right_top.append(i)
                right_top.append(j)
                return right_top

def bottom_left(img):
    left_bottom = []
    h,w = img.shape
    for i in range(h-1,0,-1):
        for j in range(w):
            if img[i][j] == 255:
                left_bottom.append(i)
                left_bottom.append(j)
                return left_bottom

def make_mask(img_shape,cor):
    points = np.array(cor)
    mask = np.zeros(img_shape)


    cv2.fillPoly(mask, pts=[points], color = 255)
    mask = (mask/255).astype('uint8')
    plt.imshow(mask, cmap='gray')
    plt.show()

    return mask

########################################################################################
## Below is the part to be changed for noisy images for better results #################
########################################################################################


def preprocess(org_img):
    img = org_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    return img


def morph(preprocess_img):
     
    img = preprocess_img.copy()

    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(img,kernel)

    subtracted_img = img - eroded_img

    coordinates = []
    print(coordinates)
    coordinates.append(top_left(subtracted_img))
    coordinates.append(top_right(subtracted_img))
    coordinates.append(bottom_right(subtracted_img))
    coordinates.append(bottom_left(subtracted_img))


    mask = make_mask(preprocess_img.shape,coordinates)
    extracted_maze = mask*(255 - preprocess_img)

    return extracted_maze


# Verified all working fine

org_img = cv2.imread('.\data\inputs\easy.jpg')
img = cv2.copyMakeBorder(org_img,3,3,3,3,cv2.BORDER_CONSTANT, None, value = (255,255,255))

pre = preprocess(img)
ext_img = morph(pre)

plt.imshow(ext_img,cmap = 'gray')
plt.show() 