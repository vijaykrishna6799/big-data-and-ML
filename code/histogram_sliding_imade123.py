#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:13:52 2022

@author: vijaykrishna
"""

import pandas as pd
import matplotlib.pyplot as plt

file=pd.read_csv('train_Image123_sliding_merged.csv')

f76=file['76']
f77=file['77']

nbins=60

plt.title('Histogram of sliding block feature vectors f76 training dataset for Image123')
plt.hist(f76, nbins, color='r', edgecolor='k')
plt.axvline(f76.mean(), color='g', linestyle='dotted', linewidth=1)
plt.text(f76.mean(), 261,'mean')
plt.show()



plt.title('Histogram of sliding block feature vectors f77 training dataset for Image 123')
plt.hist(f77, nbins, color='r', edgecolor='k')
plt.axvline(f77.mean(), color='g', linestyle='dashdot', linewidth=1)
plt.text(f77.mean(), 261,'mean')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

file1=pd.read_csv('testing_Image123_sliding_merged.csv')


f76=file1['76']
f77=file1['77']

nbins=60

plt.title('Histogram of sliding block feature vectors f76 testing dataset for Image 123')
plt.hist(f76, nbins, color='b', edgecolor='k')
plt.axvline(f76.mean(), color='b', linestyle='dashed', linewidth=1)
plt.text(f76.mean(), 261,'mean')
plt.show()


plt.title('Histogram of sliding block feature vectors f77 testing dataset for Image 123')
plt.hist(f77, nbins, color='b', edgecolor='k')
plt.axvline(f77.mean(), color='g', linestyle='dashed', linewidth=1)
plt.text(f77.mean(), 261,'mean')
plt.show()