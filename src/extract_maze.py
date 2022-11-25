import numpy as np
import cv2
from matplotlib import pyplot as plt
from prep_maze import *

def make_mask(img_shape,cor):
    points = np.array(cor)
    mask = np.zeros(img_shape)

    cv2.fillPoly(mask, pts=[points], color = 255)
    mask = (mask/255).astype('uint8')

    return mask


def order_points_new(pts):
    pts = np.array(pts)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype=int)


def connectedComp(img):
    img = img.copy()
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,0)
    h = img.shape[0]
    w = img.shape[1]
    l_img = np.zeros((h,w))
    vis = np.zeros(h*w,dtype = 'int')
    label = 1
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(img[i][j]==255 and l_img[i][j] == 0):
                queue = []
                queue.append(i*w+j)
                vis[i*w+j] = 1
                while len(queue) != 0:
                    temp = queue.pop(0)
                    x = int(temp/w)
                    y = temp%w
                    l_img[x][y] = label
                    for l in range(-1,2):
                        for m in range(-1,2):
                            if(vis[(x+l)*w +y+m] == 0 and img[x+l][y+m] == 255):
                                queue.append((x+l)*w +y+m)
                                vis[(x+l)*w +y+m] = 1
                label += 1
    l_img = l_img[1:h-1,1:w-1]
    return l_img



def corner_extraction(list_point):
    
    list_pointx = np.array(list_point.copy())
    list_pointy = np.array(list_point.copy())

    arr1inds = list_pointx[:,0].argsort()
    list_pointx[:,1] = list_pointx[:,1][arr1inds[::1]]
    list_pointx[:,0] = list_pointx[:,0][arr1inds[::1]]

    arr1inds = list_pointy[:,1].argsort()
    list_pointy[:,1] = list_pointy[:,1][arr1inds[::1]]
    list_pointy[:,0] = list_pointy[:,0][arr1inds[::1]]

    xmin = list_pointx[0][0]
    xmax = list_pointx[list_pointx.shape[0]-1][0]

    ymin = list_pointy[0][1]
    ymax = list_pointy[list_pointy.shape[0]-1][1]

    # x min
    temp_xmin = []
    for x in list_pointx:
        if(x[0] > xmin):
            break

        temp_xmin.append(x[1])

    temp_xmin.sort()
    xmin_y1 = temp_xmin[0]
    xmin_y2 = temp_xmin[len(temp_xmin)-1]

    # x max
    temp_xmax = []
    for x in range(list_pointx.shape[0]-1,-1,-1):
        if(list_pointx[x][0] < xmax):
            break

        temp_xmax.append(list_pointx[x][1])

    temp_xmax.sort()
    xmax_y1 = temp_xmax[0]
    xmax_y2 = temp_xmax[len(temp_xmax)-1]

    # y min
    temp_ymin = []
    for y in list_pointy:
        if(y[1] > ymin):
            break

        temp_ymin.append(y[0])

    temp_ymin.sort()
    ymin_x1 = temp_ymin[0]
    ymin_x2 = temp_ymin[len(temp_ymin)-1]

    # y max
    temp_ymax = []
    for y in range(list_pointy.shape[0]-1,-1,-1):
        if(list_pointy[y][1] < ymax):
            break

        temp_ymax.append(list_pointy[y][0])

    temp_ymax.sort()
    ymax_x1 = temp_ymax[0]
    ymax_x2 = temp_ymax[len(temp_ymax)-1]

    corners = []
    corners.append([xmin,xmin_y1])
    corners.append([xmin,xmin_y2])
    corners.append([xmax,xmax_y1])
    corners.append([xmax,xmax_y2])
    corners.append([ymin_x1,ymin])
    corners.append([ymin_x2,ymin])
    corners.append([ymax_x1,ymax])
    corners.append([ymax_x2,ymax])

    return corners

# Manhattan distance
def dist(pt1, pt2):
    return np.abs(pt1[0]-pt2[0]) + np.abs(pt1[1]-pt2[1])

def unique_corners(corners):

    crns = []

    crns.append(corners[0])

    for c1 in corners:
        flag=1
        for c2 in crns:
            if(dist(c1,c2)<=20):
                flag=0
        
        if(flag==1):
            crns.append(c1)
    
    return crns

def extract_maze(preprocessed_img):

    binary_img = preprocessed_img.copy()

    label_img = connectedComp(binary_img)

    h,w = label_img.shape
    new = np.zeros((h,w),dtype = 'uint8')

    for i in range(h):
        for j in range(w):
            if label_img[i][j] == 1:
                new[i][j] = 255

    num_of_labels = int(np.max(label_img))
    count_label = np.zeros((num_of_labels+1,2),dtype = int)
    for i in range(1,num_of_labels+1):
        count_label[i][1] = np.count_nonzero(label_img == i)
        count_label[i][0] = i

    arr1inds = count_label[:,1].argsort()
    count_label[:,0] = count_label[:,0][arr1inds[::-1]]
    count_label[:,1] = count_label[:,1][arr1inds[::-1]]

    top1,top2 = count_label[0][0],count_label[1][0]

    points_list = list()

    for i in range(h):
        for j in range(w):
            if label_img[i][j] == top1 or label_img[i][j] == top2:
                temp = [i,j]
                points_list.append(temp)

    corner = corner_extraction(points_list)

    refined_corner = unique_corners(corner)
    for i in range(4):
        temp = refined_corner[i][0]
        refined_corner[i][0] = refined_corner[i][1]
        refined_corner[i][1] = temp
    
    refined_corner = order_points_new(refined_corner)
    #print(refined_corner)
    mask = make_mask(label_img.shape,refined_corner)

    extracted_img = mask*preprocessed_img
    plt.imshow(255 - extracted_img,cmap = 'gray')
    plt.show()
    
    # plt.imshow(preprocessed_img_copy-preprocessed_img,cmap = 'gray')
    # plt.show()

    return (255 - extracted_img),refined_corner


# Verified all working fine

# img = cv2.imread('/home/prash/Documents/DIP_Project_new/data/inputs/noisy.jpg')

# pre = preprocess(img)
# extract_maze(pre)
# plt.imshow(ext_img,cmap = 'gray')
# plt.show()