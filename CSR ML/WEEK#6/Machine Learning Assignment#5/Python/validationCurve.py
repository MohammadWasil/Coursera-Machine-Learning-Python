# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:08:33 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
import matplotlib.pyplot as plot

import trainLinearRegression as tlr
import linearRegCostFunction as lrcf

def ValidationCurve(X,y,Xval,yval):
    
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros_like(lambda_vec)
    error_val = np.zeros_like(lambda_vec)
    
    for i in range(0,lambda_vec.shape[0]):
        
        lambda_=lambda_vec[i]
        theta = tlr.TrainLinearRegression(X, y, lambda_)
        error_train[i] = lrcf.linearRegCostFunction(X, y, theta, 0)[0]
        error_val[i] = lrcf.linearRegCostFunction(Xval, yval, theta, 0)[0]

    return lambda_vec, error_train, error_val
    