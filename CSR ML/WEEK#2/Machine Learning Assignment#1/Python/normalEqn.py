# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 22:39:41 2018

@author: Mohammad Wasil Saleem
"""

from numpy.linalg import pinv

def NormalEquation(X, y):
    
    inverse = X.T.dot(X)
    temp = pinv(inverse)
    trans = temp.dot(X.T)
    theta = trans.dot(y)
    return theta
