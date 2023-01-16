#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:52:13 2022

@author: vijaykrishna
"""

import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('train_Image123_merged.csv')

f76=file['76']
f77=file['77']

nbins=60

plt.title('Histogram of block feature vectors f76 training dataset for Image123')
plt.hist(f76, nbins, color='r', edgecolor='k')
plt.axvline(f76.mean(), color='g', linestyle='dotted', linewidth=1)
plt.text(f76.mean(), 261,r'mean')
plt.show()



plt.title('Histogram of block feature vectors f77 training dataset for Image 123')
plt.hist(f77, nbins, color='r', edgecolor='k')
plt.axvline(f77.mean(), color='g', linestyle='dashdot', linewidth=1)
plt.text(f77.mean(), 261,r'mean')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('testing_Image123_merged.csv')


f76=file['76']
f77=file['77']

nbins=60

plt.title('Histogram of block feature vectors f76 testing dataset for Image123')
plt.hist(f76, nbins, color='b', edgecolor='k')
plt.axvline(f76.mean(), color='g', linestyle='solid', linewidth=1)
plt.text(f76.mean(), 261,r'mean')
plt.show()


plt.title('Histogram of block feature vectors f77 testing dataset for Image123')
plt.hist(f77, nbins, color='b', edgecolor='k')
plt.axvline(f77.mean(), color='g', linestyle='dotted', linewidth=1)
plt.text(f77.mean(), 261,r'mean')
plt.show()