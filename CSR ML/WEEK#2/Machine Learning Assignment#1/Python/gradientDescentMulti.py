# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 23:30:52 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
import computeCostMulti as ccm

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1));
    
    for i in range(num_iters):
        #h = theta[0] + (theta[1] * X[1])
        #summation = (h[1] - y).sum()
        #temp1 = theta[0] - (alpha * (summation / m))
        #temp2 = theta[1] - (alpha * ((summation * X[1]) / m))
        #theta[0] = temp1
        #theta[1] = temp2
               
        h = X.dot(theta)
        error = h-y
        mul = X.T.dot(error)
        temp = theta - (alpha * (mul / m) ) 
        theta = temp
        #theta = temp
        # To display how much cost is reducing at every iteration.
        J_history[i] = ccm.computeCostMulti(X, y, theta);  
        #print(J_history[i])

    return (theta, J_history)
    