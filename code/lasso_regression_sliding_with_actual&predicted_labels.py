#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 21:34:11 2022

@author: vijaykrishna
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv('train_Image123_sliding_merged.csv' , header = None, index_col=None)
df2 = pd.read_csv('actual_and_predicted_training123_sliding.csv', header= None, index_col=None)

print(df1)
print(df2.shape)

 
# concatenating df3 and df4 along columns
horizontal_concat = pd.concat([df1, df2], axis=1, ignore_index=True)
print(horizontal_concat)
 
horizontal_concat.to_csv('actual_and_predicted_training123_sliding_concat.csv')

print(horizontal_concat.shape)
print(horizontal_concat)

#train_df = pd.concat(train_class_df_list, ignore_index=True)


#to drop column

horizontal_concat = horizontal_concat.drop(0, axis=1)
print(horizontal_concat.shape)
print(horizontal_concat)

#horizontal_concat = horizontal_concat.drop(83, axis=1)
#df3=horizontal_concat
#final=pd.concat(df3, ignore_index=True)
#horizontal_concat = horizontal_concat.drop(1, axis=1)
print(horizontal_concat.shape)

horizontal_concat = horizontal_concat.drop(0, axis=0).reset_index(0)
print(horizontal_concat)
print(horizontal_concat.shape)

#horizontal_concat.rename(columns = {'84':'83'}, inplace = True)

#horizontal_concat = horizontal_concat.drop(index, axis=1)

horizontal_concat.to_csv('training123_sliding_with_predicted_label.csv',index=False)

