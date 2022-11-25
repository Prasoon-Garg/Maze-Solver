import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# from skimage import measure
# from skimage.measure import regionprops
# from prep_maze import preprocess


def gate_find(preprocess_img):

    img = preprocess_img.copy()
    img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0,0,0])
    m, n = img.shape

    k1 = np.zeros(m)
    for i in range(m):
        for j in range(n):
            if img[i,j] == 255:
                k1[i] = j
                break
            elif j == n-1:
                k1[i] = j
    l1 = np.zeros(m)
    for i in range(1, k1.shape[0]-1, 2):
        l1[i] = k1[i+1] - k1[i-1]
    l1 = np.abs(l1)
    p1 = l1[l1 > 10]
    r1 = len(p1)

    k2 = np.zeros(m)
    for i in range(m):
        for j in range(n-1, -1, -1):
            if img[i,j] == 255:
                k2[i] = j
                break
            elif j == 0:
                k2[i] = j
    l2 = np.zeros(m)
    for i in range(1, k2.shape[0]-1, 2):
        l2[i] = k2[i+1] - k2[i-1]
    l2 = np.abs(l2)
    p2 = l2[l2 > 10]
    r2 = len(p2)

    k3 = np.zeros(n)
    for i in range(n):
        for j in range(m):
            if img[j,i] == 255:
                k3[i] = j
                break
            elif j == m-1:
                k3[i] = j
    l3 = np.zeros(n)
    for i in range(1, k3.shape[0]-1, 2):
        l3[i] = k3[i+1] - k3[i-1]
    l3 = np.abs(l3)
    p3 = l3[l3 > 10]
    r3 = len(p3)

    k4 = np.zeros(n)
    for i in range(n):
        for j in range(m-1, -1, -1):
            if img[j,i] == 255:
                k4[i] = j
                break
            elif j == 0:
                k4[i] = j
    l4 = np.zeros(n)
    for i in range(1, k4.shape[0]-1, 2):
        l4[i] = k4[i+1] - k4[i-1]
    l4 = np.abs(l4)
    p4 = l4[l4 > 10]
    r4 = len(p4)
    
    l = np.array([l1, l2, l3, l4])
    r = np.array([r1, r2, r3, r4])
    r = np.argsort(r)
    l = l[r[::-1]]
    l = l[0:2]
    g = []
    d = 1

    if np.array_equal(l[0], l1):
        l1 = np.argsort(l1)
        l1 = l1[-r1:]
        y = np.mean(l1[0:2]).astype(int)
        x = ((k1[l1[0]-d]+k1[l1[1]+d])/2).astype(int)
        print("Gate 1: ", x, y)
        g.append([y, x])
    elif np.array_equal(l[0], l2):
        l2 = np.argsort(l2)
        l2 = l2[-r2:]
        y = np.mean(l2[0:2]).astype(int)
        x = ((k2[l2[0]-d]+k2[l2[1]+d])/2).astype(int)
        print("Gate 2: ", x, y)
        g.append([y, x])
    elif np.array_equal(l[0], l3):
        l3 = np.argsort(l3)
        l3 = l3[-r3:]
        x = np.mean(l3[0:2]).astype(int)
        y = ((k3[l3[0]+d]+k3[l3[1]-d])/2).astype(int)
        print("Gate 3: ", x, y)
        g.append([y, x])
    elif np.array_equal(l[0], l4):
        l4 = np.argsort(l4)
        l4 = l4[-r4:]
        x = np.mean(l4[0:2]).astype(int)
        y = ((k4[l4[0]+d]+k4[l4[1]-d])/2).astype(int)
        print("Gate 4: ", x, y)
        g.append([y, x])
        
    if np.array_equal(l[1], l1):
        l1 = np.argsort(l1)
        l1 = l1[-r1:]
        y = np.mean(l1[0:2]).astype(int)
        x = ((k1[l1[0]+d]+k1[l1[1]-d])/2).astype(int)
        print("Gate 5: ", x, y)
        g.append([y, x])
    elif np.array_equal(l[1], l2):
        l2 = np.argsort(l2)
        l2 = l2[-r2:]
        y = np.mean(l2[0:2]).astype(int)
        x = ((k2[l2[0]+d]+k2[l2[1]-d])/2).astype(int)
        print("Gate 6: ", x, y)
        g.append([y, x])
    elif np.array_equal(l[1], l3):
        l3 = np.argsort(l3)
        l3 = l3[-r3:]
        x = np.mean(l3[0:2]).astype(int)
        y = ((k3[l3[0]-d]+k3[l3[1]+d])/2).astype(int)
        g.append([y, x])
    elif np.array_equal(l[1], l4):
        l4 = np.argsort(l4)
        l4 = l4[-r4:]
        x = np.mean(l4[0:2]).astype(int)
        y = ((k4[l4[0]-d]+k4[l4[1]+d])/2).astype(int)
        print("Gate 8: ", x, y)
        g.append([y, x])
    
    g = np.abs(g - np.array([[10, 10],[10,10]]))
    return g

# img = cv.imread('.\data\inputs\easy.jpg')
# prep_img = preprocess(img)
# g = gate_find(prep_img)
# print(g)
# cv.circle(img, (g[0][1], g[0][0]), 5, (0, 0, 255), -1)
# cv.circle(img, (g[1][1], g[1][0]), 5, (0, 0, 255), -1)
# plt.imshow(img)
# plt.show()