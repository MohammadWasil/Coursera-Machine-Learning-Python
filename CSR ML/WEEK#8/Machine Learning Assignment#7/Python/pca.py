# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:12:08 2018

@author: Mohammad Wasil Saleem
"""

from numpy.linalg import svd
import numpy as np

def PCA(X):
    [m,n] = X.shape
    sigma = (1/m)*(X.T.dot(X))
    [U,S,V] = svd(sigma)
    S=np.diag(S)                #converting to diagonal matrix
    #print(S.shape)
    return U,S