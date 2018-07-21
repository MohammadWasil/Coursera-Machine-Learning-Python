# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 01:12:21 2018

@author: Mohammad Wasil Saleem
"""

def computeCostMulti(X, y, theta):
    # j = 1/2m * summation (h(x) - y )^2m, summation from i =1 to m.
    m = len(y)
    h = X.dot(theta)
    summation = ((h - y)**2).sum() 
    J = summation / (2 * m)
    return J