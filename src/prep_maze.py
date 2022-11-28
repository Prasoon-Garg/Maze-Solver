import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(org_img):
    img = org_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    return img