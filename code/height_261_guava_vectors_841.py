#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:24:19 2022

@author: vijaykrishna
"""


import numpy as np
import pandas as pd
import cv2

#Loading the grey guava image
Grey_fruit2 = cv2.imread('Grey_image_of_guava.jpg')
#printing the original dimensions of the image
print("Original dimensions of Guava image : ", Grey_fruit2.shape)
std_height = 261
height = Grey_fruit2.shape[0]
width = Grey_fruit2.shape[1]

#let us give the resize function

def width_div9(height, width):
    print("Original Height :", height)
    print("Original width :", width)
    aspect_ratio = round(height/width)
    new_fruit2_height = aspect_ratio * 261
    print("Setting the Height fixed to :" , new_fruit2_height)
    
    
    return round(new_fruit2_height)

#function for dividing by 9
def dimension_div9(dimensions_value):
    remainder = dimensions_value % 9
    if remainder == 0:
        dimensions_value = dimensions_value
#dividing by 9
    else:
        dimensions_value = dimensions_value + (9-remainder)
        
    return dimensions_value

#setting the width to be divisible by 9
new_fruit2_width = width_div9(height,width)
new_height = std_height
#setting appropriate values to be divisible by 9
print("New Dimension divisible by 9 :" , new_height, new_fruit2_width)
Grey_fruitU = cv2.cvtColor(Grey_fruit2, cv2.COLOR_BGR2GRAY)
image_resized = cv2.resize(Grey_fruitU, dsize = (new_height, new_fruit2_width))
cv2.imwrite('Guava_height_261_resize.jpg', image_resized)

# code for generating non-overlapping block feature vectors

generate_blocks = round((new_height) * (new_fruit2_width)/81)
print(generate_blocks)

flat_image = np.full((generate_blocks, 82), 2)
k = 0

for i in range(0 , new_height , 9):
    for j in range(0 , new_fruit2_width , 9):
        tmp = image_resized[i : i+9, j: j+9]
        print(tmp)
        flat_image[k,0:81] = tmp.flatten()
        k = k+1

feature_space = pd.DataFrame(flat_image)
feature_space.to_csv('Image3_Guava_81.csv', index=False)


