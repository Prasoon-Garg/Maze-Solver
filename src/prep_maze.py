import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(org_img):
    img = org_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    return img