#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 00:23:35 2022

@author: vijaykrishna
"""
import numpy as np
import pandas as pd
import cv2


#Loading the grey banana image
Grey_fruit = cv2.imread('Grey_image_of_banana.jpg')
#printing the original dimensions of the image
print("Original dimensions of banana image: ", Grey_fruit.shape)
std_height = 261
height = Grey_fruit.shape[0]
width = Grey_fruit.shape[1]

#let us give the resize function

def width_div9(height, width):
    print("Original Height :", height)
    print("Original width :", width)
    aspect_ratio = round(height/width)
    new_fruit_height = aspect_ratio * 261
    print("Setting the Height fixed to :" , new_fruit_height)
    
    
    return round(new_fruit_height)

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
new_fruit_width = width_div9(height,width)
new_height = std_height
#setting appropriate values to be divisible by 9
print("New Dimension divisible by 9 :" , new_height, new_fruit_width)
Grey_fruitG = cv2.cvtColor(Grey_fruit, cv2.COLOR_BGR2GRAY)
image_resized = cv2.resize(Grey_fruitG, dsize = (new_height, new_fruit_width))
cv2.imwrite('Banana_height_261_resize.jpg', image_resized)


# code for generating block feature vectors

generate_blocks = round((new_height) * (new_fruit_width)/81)
print(generate_blocks)

flat_image = np.full((generate_blocks, 82), 0)
k = 0

for i in range(0 , new_height , 9):
    for j in range(0 , new_fruit_width , 9):
        tmp = image_resized[i : i+9, j: j+9]
        print(tmp)
        flat_image[k,0:81] = tmp.flatten()
        k = k+1

#feature_space = pd.DataFrame(flat_image)
#feature_space.to_csv('Image1_banana_81.csv', index=False)












