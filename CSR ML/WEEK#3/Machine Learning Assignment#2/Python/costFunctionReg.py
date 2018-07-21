# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:30:47 2018

@author: Mohammad Wasil Saleem
"""

import costFunction as cf

import numpy as np

def costFunctionReg(theta, X, y, lambdaa):
    m = len(y)
    #theta_reg = theta
    #theta_reg[0] = 0
    
    h = np.zeros((m,1))
    h[:, 0] = cf.Sigmoid(X.dot(theta))
    
    cost1 = -((y.T)).dot(np.log(h))
    cost0 = -((1-y).T).dot(np.log(1-h))

    J = sum(cost1 + cost0 ) * (1/m)
    J_reg = (lambdaa / (2.0 * m)) * np.sum(theta[1:] ** 2)

    J_reg = J_reg + J
    
    grad = ((h-y).T.dot(X)) * (1/m)
    grad_reg = (lambdaa / m) * theta
    grad_reg[0] = 0
    
    grad_reg = grad_reg + grad
    #print(grad_reg.shape)
    return J_reg, grad_reg

def Sigmoid(x):
    # for large positive value, sigmoid will be close to 1.
    # for large negative value, sigmoid will be close to 0.  
    g = ( 1 / (1 + np.exp(-x)))
    
    return g

def Predict(X, theta):
    return Sigmoid(X.dot(theta)) >= 0.5
    #return  1 / (1 + np.exp(-(X.dot(theta)))) >= 0.5