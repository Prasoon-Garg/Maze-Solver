# import cv2 
import numpy as np
# import matplotlib.pyplot as plt

# from extract_maze import *
# from gate_find import *
# from path_find import *
# from trace_path import *

def gate_shift(gate, img_shape):
    h,w = img.shape

    temp = gate[0][0]
    gate[0][0] = gate[0][1]
    gate[0][1] = temp

    temp = gate[1][0]
    gate[1][0] = gate[1][1]
    gate[1][1] = temp
    
    if(gate[0][1] > w/2):
        gate[0][1] -= 3
    elif(gate[0][1] <= w/2):
        gate[0][0] += 3
    
    if(gate[1][0] > h/2):
        gate[1][0] -= 3
    elif(gate[1][0] <= h/2):
        gate[1][0] += 3

    if(gate[0][1] > w/2):
        gate[0][1] -= 3
    elif(gate[0][1] <= w/2):
        gate[0][1] += 3

    if(gate[1][1] > w/2):
        gate[1][1] -= 3
    elif(gate[1][1] <= w/2):
        gate[1][1] += 3

    return gate


gate = np.array([[572,0], [760,730]])

print(gate_shift(gate, (1200,900)))