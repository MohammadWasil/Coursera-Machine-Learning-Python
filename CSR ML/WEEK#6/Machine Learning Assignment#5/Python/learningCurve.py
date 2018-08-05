# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 00:15:58 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

import trainLinearRegression as tlr
import linearRegCostFunction as lrcf

def LearningCurve(X, y, Xval, yval, lambda_):
    #In this function, you will compute the train and test errors for
    #dataset sizes from 1 up to m. In practice, when working with larger
    #datasets, you might want to do this in larger intervals.
    [m, n] = X.shape
    error_train = np.zeros((m+1,1))
    error_cross_validation = np.zeros((m+1,1))
    
    for i in range(1, m+1):
        X_train = X[0:i, :]
        y_train = y[0:i]
        theta = tlr.TrainLinearRegression(X_train, y_train, lambda_)
        error_train[i] = lrcf.linearRegCostFunction(X_train, y_train, theta, lambda_)[0]
        error_cross_validation[i] = lrcf.linearRegCostFunction(Xval, yval, theta, lambda_)[0]
        
    return error_train, error_cross_validation
        
