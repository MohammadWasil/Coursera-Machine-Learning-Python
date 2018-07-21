# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 19:20:03 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def costFunction(theta, X, y):
    m = len(y)
    h = np.zeros((m,1))
    h[:, 0] = Sigmoid(X.dot(theta))
    cost1 = -((y.T).dot(np.log(h)))
    cost0 = (-((1-y).T)).dot(np.log(1-h))

    #J = (-((y.T) * log(h)) - ((1-y).T * log(1-h))) * (1/m);
    J = sum(cost1 + cost0 ) * (1/m)
    grad = (X.T.dot(h-y)) * (1/m)

    return J, grad

    
def Sigmoid(x):
    # for large positive value, sigmoid will be close to 1.
    # for large negative value, sigmoid will be close to 0.  
    g = ( 1 / (1 + np.exp(-x)))
    
    return g

def Predict(X, theta):
    return Sigmoid(X.dot(theta)) >= 0.5
    #return  1 / (1 + np.exp(-(X.dot(theta)))) >= 0.5
    