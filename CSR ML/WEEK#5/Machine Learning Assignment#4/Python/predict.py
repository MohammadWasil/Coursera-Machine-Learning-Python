# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:32:13 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

import nnCostFunction as nncf

def Predict(Theta1, Theta2, X):
    [m, n] = X.shape
    
    # You need to return the following variables correctly 
    prediction = np.zeros((m, 1))
       
    X  = np.concatenate((np.ones((m,1)), X), axis=1)
    h1 = nncf.Sigmoid(X.dot(Theta1.T))
    h2 = nncf.Sigmoid(np.concatenate((np.ones((m,1)),h1 ), axis=1).dot(Theta2.T) )
    prediction = 1 + np.argmax(h2, axis=1)
        
    return prediction
