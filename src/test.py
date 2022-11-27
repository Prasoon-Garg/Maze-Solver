import cv2 
import numpy as np
import matplotlib.pyplot as plt

from extract_maze import *
from gate_find import *
from path_find import *
from solution_overlay import *

org_img = cv2.imread('/home/prash/Documents/DIP_Project_new/data/inputs/rotated.jpg')

preprocess_img = preprocess(org_img)
extracted_maze,corners = extract_maze(preprocess_img)
gates = gate_find(preprocess_img,corners)

# print(gates)

path_res = final_path(extracted_maze,gates[0],gates[1])

final_res = sol_overlay(org_img,path_res)


plt.imshow(final_res)
plt.show()