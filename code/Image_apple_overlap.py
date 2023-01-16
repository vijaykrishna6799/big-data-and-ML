#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:14:59 2022

@author: vijaykrishna
"""

import cv2
import numpy as np
import pandas as pd

#read  the image
read_fruit = cv2.imread('apple1.jpg')
#covert apple to grayscale
fruit_gray = cv2.cvtColor(read_fruit, cv2.COLOR_BGRA2GRAY)
height , width = fruit_gray.shape
std_height = 261

def width_resize(height , width):
    print("Original Height :", height)
    print("Original Width :", width)
    aspect_ratio = width/height
    new_fruit_width = aspect_ratio * 261
    print("My width :" , new_fruit_width)
    return round(new_fruit_width)
def dimesion_div8(height , width):
      new_fruit_width = width_resize(height , width)
      remainder = new_fruit_width % 9
      if remainder == 0:
         return new_fruit_width
      else:
          new_fruit_width  = new_fruit_width - remainder
          
      return new_fruit_width
#Values when both need to be divisible by 9 
new_fruit_width = dimesion_div8(height,width)
print('Standard Height 261 dimension' , std_height , new_fruit_width)
resized_fruit = cv2.resize(fruit_gray, dsize = (new_fruit_width, std_height))
heightd , widthd = resized_fruit.shape


dd = round((heightd -7 ) * (widthd - 7))
flat_image = np.full((dd, 82), 0)

k = 0 
for i in range( 0 , heightd -7 , 1):
    for j in range(0 , widthd - 7 , 1):
        tmp = resized_fruit[ i:i + 8 , j:j + 8]
        flat_image[k, 0:64] = tmp.flatten()
        k = k + 1
        
feature_space = pd.DataFrame(flat_image)
feature_space.to_csv('Image_apple_Overlap.csv' , index=False)

