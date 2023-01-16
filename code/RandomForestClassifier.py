#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 22:09:09 2022

@author: vijaykrishna
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import time
x_train= pd.read_csv('train_Image123_sliding_merged.csv',header=None)
x_test= pd.read_csv('testing_Image123_sliding_merged.csv', header=None)

y = x_train[82]
print(y)
y.drop(0, axis=0, inplace=True)
Y = np.array(y)
print(x_train)
x_train.drop(82,axis=1,inplace=True)
x_train = x_train.drop(0, axis=1)
x_train = x_train.drop(0, axis=0)
print(x_train)

X = x_train
X1 = np.array(X)
print(X1)
print(Y.shape)
print(Y)


t0= time.time()

# Train the model
# n-estimators are for the number of trees in the forest
# accuracy reduces as the n-estimators value increases
randomFClassifier = RandomForestClassifier(random_state=0,n_estimators=100,oob_score=True, n_jobs=-1)
randomFClassifier = randomFClassifier.fit(X1, Y)

t1 = time.time() - t0
print('')
print('')

print("Time taken to run the Random forest model: ", t1)
print('')
print('')
#Testing the model using Trained results

test_y = x_test[82]
test_y.drop(0, axis=0, inplace=True)
Y_test = np.array(test_y)
x_test.drop(82,axis=1,inplace=True)
x_test = x_test.drop(0, axis=1)
x_test = x_test.drop(0, axis=0)
x_test_np = np.array(x_test)
y_predict = randomFClassifier.predict(x_test_np)
ydash_saved = pd.DataFrame(y_predict)
#ydash_saved.to_csv('Random_Forest_predicted_label_Image12_block_features.csv', index=False)

C_test = confusion_matrix(test_y, y_predict)
TP = C_test[1,1]
FN = C_test[1,0]
FP = C_test[0,1]
TN = C_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

#Qualitative measures using Confusion matrix
Accuracy = 1/(1+(FPFN/TPTN))
print('test accuracy is:', Accuracy)
Precision = 1/(1+(FP/TP))
print("test_precision is:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("test_sensitivity is:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("test_specificity is:",Specificity)

print('')
print('')
print('')
#Qualitative measures of performance calculation using sklearn metrics
print('sklearn.metrics Accuracy Output',accuracy_score(test_y, y_predict))
print('sklearn.metrics Precision Value',precision_score(test_y, y_predict, average='macro'))
print('sklearn.metrics Sensitivity Value',recall_score(test_y, y_predict, average='macro'))

#generate tree for the RandomForestClassifier
#plt.figure(figsize=(100,100))
#tree.plot_tree(randomFClassifier.estimators_[2], filled=True)
#plt.savefig('RandomForest_Image12_block_feature_vectors(1).pdf') 



    

