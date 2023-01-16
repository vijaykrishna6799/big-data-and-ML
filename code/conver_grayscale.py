#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 13:33:12 2022

@author: vijaykrishna
"""

import cv2
import matplotlib.pyplot as plt

# reading the image
banana_image = cv2.imread('banana1.jpg')
apple_image = cv2.imread('apple1.jpg')
guava_image = cv2.imread('guava1.jpg')

plt.imshow(banana_image[:,:,1])

#convert banana to the grey scale
banana_grey = cv2.cvtColor(banana_image, cv2.COLOR_BGRA2GRAY)

#writing the grey image to another file
cv2.imwrite('Grey_image_of_banana.jpg', banana_grey)


#get the dimensions
dimensions = banana_image.shape
height = banana_image.shape[0]
width = banana_image.shape[1]
channels = banana_image.shape[2]

#printing the dimensions
print("Dimensions of Banana : ")
print('Image Dimension:' ,dimensions)
print('Image height:', height)
print('Image width:', width)
print('Number of channels:', channels)

plt.imshow(apple_image[:,:,1])

#convert apple to the grey scale
apple_grey = cv2.cvtColor(apple_image, cv2.COLOR_BGRA2GRAY)

#writing the grey image to another file
cv2.imwrite('Grey_image_of_apple.jpg', apple_grey)


#get the dimensions
dimensions = apple_image.shape
height = apple_image.shape[0]
width = apple_image.shape[1]
channels = apple_image.shape[2]

#printing the dimensions
print("Dimensions of apple : ")
print('Image Dimension:' ,dimensions)
print('Image height:', height)
print('Image width:', width)
print('Number of channels:', channels)


plt.imshow(guava_image[:,:,1])

#convert guava to the grey scale
guava_grey = cv2.cvtColor(guava_image, cv2.COLOR_BGRA2GRAY)

#writing the grey image to another file
cv2.imwrite('Grey_image_of_guava.jpg', guava_grey)


#get the dimensions
dimensions = guava_image.shape
height = guava_image.shape[0]
width = guava_image.shape[1]
channels = guava_image.shape[2]

#printing the dimensions
print("Dimensions of guava : ")
print('Image Dimension:' ,dimensions)
print('Image height:', height)
print('Image width:', width)
print('Number of channels:', channels)



