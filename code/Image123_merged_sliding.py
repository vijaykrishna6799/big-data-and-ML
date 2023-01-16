#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:59:13 2022

@author: vijaykrishna
"""


import pandas as pd
import numpy as np

image1_sliding_banana = pd.read_csv('banana1_sliding_block_vectors.csv')
image2_sliding_apple = pd.read_csv('apple1_sliding_block_vectors.csv')

frames = [image1_sliding_banana, image2_sliding_apple]

merged = pd.concat(frames)

index = np.arange(len(merged))
merge = np.random.permutation(index)
merge = merged.sample(frac=1).reset_index(drop=True)
merge.to_csv('Image12_sliding_banana_apple_merged.csv', index=False)




image3_sliding_banana_apple = pd.read_csv('Image12_sliding_banana_apple_merged.csv')
image3_sliding_guava = pd.read_csv('guava1_sliding_block_vectors.csv')
frames = [image3_sliding_banana_apple, image3_sliding_guava]

merged = pd.concat(frames)

index = np.arange(len(merged))
merge = np.random.permutation(index)
merge = merged.sample(frac=1).reset_index(drop=True)
merge.to_csv('Image123_sliding_banana_apple_guava_merged.csv', index=False)

