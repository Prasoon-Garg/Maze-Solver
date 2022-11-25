import cv2 
import numpy as np
import matplotlib.pyplot as plt

from extract_maze import *
from gate_find import *
from path_find import *
from trace_path import *

org_img = cv2.imread('.\data\inputs\easy.jpg')
org_img = cv2.copyMakeBorder(org_img,3,3,3,3,cv2.BORDER_CONSTANT, None, value = (255,255,255))
preprocess_img = preprocess(org_img)
extracted_maze = morph(preprocess_img)
gates = gate_find(preprocess_img)

# gates[0][1] = 110
# gates[1][1] = 490

# gates[0][0] = 490
# gates[1][0] = 110

print(gates)

path_res = final_path(extracted_maze,gates[0],gates[1])
print(path_res)
final_res = sol_overlay(org_img,path_res)
print(final_res.shape)

plt.imshow(final_res)
plt.show()