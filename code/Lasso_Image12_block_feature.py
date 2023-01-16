#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 22:20:47 2022

@author: vijaykrishna
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

#read the training and the testing dataset
x_train = pd.read_csv('train_Image12_merged.csv' , header = None)
print('initial shape of x_train is :',x_train.shape)
x_train = x_train.drop(0, axis=1)
x_train = x_train.drop(0, axis=0)

print('shape of x_train after dropping unwanted 0th column :',x_train.shape)
print(x_train)

# train_Image123_sliding_merged.csv has 82 columns(0-81) including the indexing column
x_test = pd.read_csv('testing_Image12_merged.csv',header=None)
print('shape of x_test is :',x_test.shape)
x_test = x_test.drop(0, axis=1)
x_test = x_test.drop(0, axis=0)

print('shape of x_test after dropping the index is :',x_test.shape)

#read the label and store in y
y = x_train[82] # last but one column is 81
Y = np.array(y)
print('array of Y is:',Y) #Y is array of last column
print('shape of Y is:', Y.shape)

#drop the label column 
x_train.drop(82,axis=1,inplace=True)
print('shape of x_train is :',x_train.shape)
X = x_train
X1 = np.array(X)
#lambda value is set to 0.01
lamda = 0.01

# X' is the transpose of X
X2 = X1.transpose()
print('shape of X2 is:', X2.shape)
#XX transpose is by multiplying X1 and X2
XX_transpose = np.matmul( X2 , X1)
#shape of XX after multiplying X1 and X2 is (81,81)
print('shape of XX_transpose is :',XX_transpose.shape)
# [XX']inverse is IXX
IXX = inv(XX_transpose)
print('shape of inverse of XX_transpose is:',IXX.shape)
#Ydash = Y.transpose()
ymulX = np.matmul(X2, Y)
print(ymulX.shape)

first_par = np.matmul(ymulX,IXX)
S= np.sign(first_par)


second_argument = (S*(lamda/2))
sub_value = ymulX-second_argument
A = np.matmul(IXX , sub_value)

y_dash = np.matmul(X1, A)
ZZ2 = y_dash > y_dash.mean()
y_dash_training = ZZ2.astype(int)

# Save the predicted values
y_dash_training_saved = pd.DataFrame(y_dash_training)
#y_dash_training_saved.to_csv('actual_and_predicted_training12.csv',index=False)
print(y_dash_training_saved.shape)

print('Y is :', Y)
print('y_dash training is:',y_dash_training)
#Confusion matrix
C_train = confusion_matrix(Y, y_dash_training)
TN = C_train[0,0]
FP = C_train[0,1]
FN = C_train[1,0]
TP = C_train[1,1]
FPFN = FP+FN
TPTN = TP+TN


Accuracy = 1/(1+(FPFN/TPTN))
print("Accuracy for trainig set is:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Precision values for training set is:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Sensitivity of training set is:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Specificity for training dataset is:",Specificity)
print('')
print('')


#Inbuilt performance metrics
print('sklearn.metrics Accuracy',accuracy_score(Y, y_dash_training))
print('sklearn.metrics precision',precision_score(Y, y_dash_training, average='macro'))
print('sklearn.metrics sensitivity',recall_score(Y, y_dash_training , average='macro'))


#Testing the model
test_y = x_test[82]
Y_test = np.array(test_y)
x_test.drop(82,axis=1,inplace=True)
x_test_np = np.array(x_test)

Z1_test = np.matmul(x_test_np, A)
Z2_test = Z1_test > Z1_test.mean()

# Test data accuracy
yhat_test = Z2_test.astype(int)
yhat_saved = pd.DataFrame(yhat_test)
#yhat_saved.to_csv('LassoTest_Image12_merged_result.csv', index=False)
print(yhat_saved.shape)
C_test = confusion_matrix(test_y, yhat_test)
TN = C_test[0,0]
FP = C_test[0,1]
FN = C_test[1,0]
TP = C_test[1,1]
FPFN = FP+FN
TPTN = TP+TN
print('Confusion Matrix is :', C_test)
Accuracy = 1/(1+(FPFN/TPTN))
print("Accuracy for testing set is:",Accuracy)
Precision = 1/(1+(FP/TP))
print("Precision values for testing set is:",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Sensitivity of testing set is:",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Specificity for testing dataset is:",Specificity)
print('')
print('')


#Inbuilt performance metrics
print('sklearn.metrics Accuracy',accuracy_score(test_y, yhat_test ))
print('sklearn.metrics precision',precision_score(test_y, yhat_test, average='macro'))
print('sklearn.metrics sensitivity',recall_score(test_y, yhat_test , average='macro'))



