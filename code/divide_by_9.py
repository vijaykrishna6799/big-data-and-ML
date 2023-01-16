#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 22:10:26 2022

@author: vijaykrishna
"""



import cv2

# read the image
GreyBanana = cv2.imread('Grey_image_of_banana.jpg')
#print the original dimensions
print("Original Dimensions of banana : ", GreyBanana.shape)

height = (GreyBanana.shape[0])/2
width = (GreyBanana.shape[1])/2

print("Halfed height :" , height)
print("Halfed width : ", width)

#function for dividing by 9
def dimension_div9(dimensions_value):
    remainder = dimensions_value % 9
    if remainder == 0:
        dimensions_value = dimensions_value
#dividing by 9
    else:
        dimensions_value = dimensions_value + (9-remainder)
        
    return dimensions_value


new_width = round(dimension_div9(height))
new_height = round(dimension_div9(width))

#height and width to be divisible by 9
print("New Dimensions divisible by 9 :", new_height,new_width)
image_resized = cv2.resize(GreyBanana, dsize = (new_height, new_width))
cv2.imwrite('Banana_imagedimension_div9.jpg', image_resized) 







# read the image for resizing the apple image
GreyApple = cv2.imread('Grey_image_of_apple.jpg')
#print the original dimensions
print("Original Dimensions of Apple : ", GreyApple.shape)

height = (GreyApple.shape[0])/2
width = (GreyApple.shape[1])/2

print("Halfed height :" , height)
print("Halfed width : ", width)

#function for dividing by 9
def dimension_div9(dimensions_value):
    remainder = dimensions_value % 9
    if remainder == 0:
        dimensions_value = dimensions_value
#dividing by 9
    else:
        dimensions_value = dimensions_value + (9-remainder)
        
    return dimensions_value


new_width = round(dimension_div9(height))
new_height = round(dimension_div9(width))

#height and width to be divisible by 9
print("New Dimensions divisible by 9 :", new_height,new_width)
image_resized = cv2.resize(GreyApple, dsize = (new_height, new_width))
cv2.imwrite('Apple_imagedimension_div9.jpg', image_resized) 







# read the image for resizing the guava image
GreyGuava = cv2.imread('Grey_image_of_guava.jpg')
#print the original dimensions
print("Original Dimensions of Guava : ", GreyGuava.shape)

height = (GreyGuava.shape[0])/2
width = (GreyGuava.shape[1])/2

print("Halfed height :" , height)
print("Halfed width : ", width)

#function for dividing by 9
def dimension_div9(dimensions_value):
    remainder = dimensions_value % 9
    if remainder == 0:
        dimensions_value = dimensions_value
#dividing by 9
    else:
        dimensions_value = dimensions_value + (9-remainder)
        
    return dimensions_value


new_width = round(dimension_div9(height))
new_height = round(dimension_div9(width))

#height and width to be divisible by 9
print("New Dimensions divisible by 9 :", new_height,new_width)
image_resized = cv2.resize(GreyGuava, dsize = (new_height, new_width))
cv2.imwrite('Guava_imagedimension_div9.jpg', image_resized) 