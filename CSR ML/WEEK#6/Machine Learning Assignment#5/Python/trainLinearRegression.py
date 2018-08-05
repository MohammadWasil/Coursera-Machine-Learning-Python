# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 21:26:52 2018

@author: Mohammad Wasil Saleem
"""
from scipy import optimize
import numpy as np

import linearRegCostFunction as lrcf

def TrainLinearRegression(X, y, lambda_):

    # Create "short hand" for the cost function to be minimized
    initial_theta = np.zeros((X.shape[1], 1))
    costFunction = lambda cf : lrcf.linearRegCostFunction(X, y, cf, lambda_)
    
    result = optimize.minimize( costFunction, initial_theta, method = 'CG', jac = True, options={ 'maxiter' : 200, 'disp' : False} )
    theta = result.x
    
    return theta