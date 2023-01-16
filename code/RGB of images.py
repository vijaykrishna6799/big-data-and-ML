#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 12:46:35 2022

@author: vijaykrishna
"""
import cv2
import matplotlib.pyplot as plt

# reading the image
banana_image = cv2.imread('banana1.jpg')
apple_image = cv2.imread('apple1.jpg')
guava_image = cv2.imread('guava1.jpg')

#print the dimensions of the image

print (banana_image.shape)
print (apple_image.shape)
print (guava_image.shape)

#RGB for banana
plt.imshow(banana_image[:,:,0])
plt.show()
plt.imshow(banana_image[:,:,1])
plt.show()
plt.imshow(banana_image[:,:,2])
plt.show()
#RGB for apple
plt.imshow(apple_image[:,:,0])
plt.show()
plt.imshow(apple_image[:,:,1])
plt.show()
plt.imshow(apple_image[:,:,2])
plt.show()
#RGB for Guava
plt.imshow(guava_image[:,:,0])
plt.show()
plt.imshow(guava_image[:,:,1])
plt.show()
plt.imshow(guava_image[:,:,2])
plt.show()