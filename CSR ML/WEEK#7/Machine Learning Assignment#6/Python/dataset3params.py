# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 20:42:05 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
from sklearn.svm import libsvm

def DataSet3Params(X,y,Xval,yval):
    C = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    
    previousError = 2000
    
    for i in range(len(C)):
        currentC = C[i]
        for j in range(len(sigma)):
            
            currentSigma = sigma[j]
            gamma = 1 / (2 * ( currentSigma ** 2 ))
            
            model = libsvm.fit(X, y, kernel = 'rbf', C=currentC, gamma=gamma)   # Train
            
            predictions = libsvm.predict(Xval, support = model[0], SV = model[1], nSV = model[2], sv_coef = model[3], 
            intercept = model[4], probA = model[5], probB = model[6], kernel = 'rbf', gamma = gamma)
            
            currentError = np.mean(predictions != yval)
            
            if currentError < previousError:
                bestC = currentC 
                bestSigma = currentSigma
                previousError = currentError
    
    
    C = bestC
    sigma = bestSigma
    
    return C, sigma
