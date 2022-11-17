import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from extract_maze import *
from gate_find import *
from path_find import *
from solution_overlay import *

org_img = cv2.imread('easy.jpg')

preprocess_img = preprocess(org_img)
extracted_maze = morph(preprocess_img)
gates = gate_find(preprocess_img)

gates[0][1] = 110
gates[1][1] = 490

gates[0][0] = 490
gates[1][0] = 110

path_res = final_path(extracted_maze,gates[0],gates[1])

final_res = sol_overlay(org_img,path_res)

plt.imshow(final_res)
plt.show()