# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:31:37 2018

@author: Mohammad Wasil Saleem
"""
import numpy as np

def linearRegCostFunction(X, y, theta, lambda_):
    [m, n] = X.shape
    h = X.dot(theta)            # 12X1 

    J = sum((h-y)**2) * (1 / (2 * m))   
    J_reg =  (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    
    J=J+J_reg

    grad = (1/m)* X.T.dot((h-y))
    grad_reg = (lambda_/m) * theta
    grad_reg[0] = 0
    
    grad = grad+grad_reg
    
    return J, grad