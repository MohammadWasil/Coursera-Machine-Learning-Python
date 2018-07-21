# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:55:54 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

import computeCost as cc

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1));
    
    for i in range(iterations):
        #h = theta[0] + (theta[1] * X[1])
        #summation = (h[1] - y).sum()
        #temp1 = theta[0] - (alpha * (summation / m))
        #temp2 = theta[1] - (alpha * ((summation * X[1]) / m))
        #theta[0] = temp1
        #theta[1] = temp2
               
        h = X.dot(theta)
        error = h-y
        mul = error.dot(X)
        theta = theta - (alpha * (mul / m) ) 
        # To display how much cost is reducing at every iteration.
        J_history[i] = cc.computeCost(X, y, theta);  
        #print(J_history[i])

    return theta