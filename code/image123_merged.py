#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 00:40:31 2022

@author: vijaykrishna
"""

import pandas as pd
import numpy as np

image1_banana = pd.read_csv('Image12_merged.csv')
image2_apple = pd.read_csv('Image3_Guava_81.csv')

frames = [image1_banana, image2_apple]

merged = pd.concat(frames)

index = np.arange(len(merged))
merge = np.random.permutation(index)
merge = merged.sample(frac=1).reset_index(drop=True)
merge.to_csv('Image123_merged.csv', index=False)
