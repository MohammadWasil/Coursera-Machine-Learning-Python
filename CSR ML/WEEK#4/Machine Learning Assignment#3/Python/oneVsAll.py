# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 20:06:32 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
from scipy import optimize

def predictOneVsAll(all_theta, X):
    pred = sigmoid(X.dot(all_theta.T))
    prediction = 1 + np.argmax(pred, axis=1)
    return prediction

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cost_function_reg(theta, X, y, lambda_):
    
    m = X.shape[0]

    h = sigmoid(X.dot(theta))
    cost = sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))
    grad = X.T.dot(h - y)
    
    cost = cost/m
    gradient = grad/m
    
    reg_cost = (lambda_ / (2.0 * m)) * np.sum(theta[1:] ** 2)
    reg_gradient = (lambda_ / m) * theta
    reg_gradient[0] = 0
    return cost + reg_cost, gradient + reg_gradient

def OneVsAll(X, y, num_labels, lambda_):
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n))
    for c in range(1, num_labels + 1):
        initial_theta = np.zeros(n)
        target = np.vectorize(int)(y == c)
        result = optimize.minimize(
            cost_function_reg,
            #cost_function_reg,
            initial_theta,
            args=(X, target, lambda_),
            method='CG',
            jac=True,
            options={
                'maxiter': 50,
                'disp': False,
            }
        )
        theta = result.x
        cost = result.fun
        print('Training theta for label %d | cost: %f' % (c, cost))
        all_theta[c - 1, :] = theta
    return all_theta
