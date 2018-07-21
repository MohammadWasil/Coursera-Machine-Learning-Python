# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:09:04 2018

@author: Mohammad Wasil Saleem
"""
import numpy as np
from scipy.io import loadmat

import displayData as dd
import lrCostFunction as lrcf
import oneVsAll as ova

# Dataset.
# 5000 training examples of handwritten digits.
# where each training example is a 20 pixel by 20 pixel grayscale image of the digit. 
# Each pixel is represented by a foating point number indicating the grayscale intensity at
# that location. 
# The 20 by 20 grid of pixels is "unrolled" into a 400-dimensional vector(Row vector).
# Each of these training examples becomes a single row in our data matrix X. 
# This gives us a 5000 by 400 matrix X where every row is a training 
# example for a handwritten digit image.

# load the data mat.
data = loadmat('D:\ML\ML\CSR ML\WEEK#4\Machine Learning Assignment#3\Python\ex3data1.mat')
X = data['X']
y = data['y'].flatten()
m = len(y)

#Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]         # Size of sel = (100, 400)

dd.displayData(sel)

# Vectorize Logistic Regression 
# Testing values.
# Testing to see if the cost function is fully vectorised or not.
theta_t = np.array([-2,-1,1,2])

onesArray = np.ones((5,1), dtype = np.int64 )
arr = (np.arange(1,16).reshape((3,5)).T)/10
X_t = np.hstack((onesArray, arr))

y_t = np.array([[1,0,1,0,1]]).T >= 0.5

lambda_t = 3

J_t, grad_t = lrcf.LRCostFunction(theta_t, X_t, y_t, lambda_t)
#J_t, grad_t = ova.cost_function_reg(theta_t, X_t, y_t, lambda_t)
print('Cost: ', J_t[0])
print('Expected cost: 2.534819')
print('Gradients: ', grad_t[0:5])
print('Expected gradients: 0.146561, -0.548558, 0.724722, 1.398003\n')

# OneVsAll training
lambdaa = 0.1
num_labels = 10
X = np.concatenate((np.ones((m, 1)), X), axis=1)
all_theta = ova.OneVsAll(X, y, num_labels, lambdaa)

prediction = ova.predictOneVsAll(all_theta, X)
accuracy = np.mean(prediction == y) * 100
print('Training Set accuracy', accuracy)
