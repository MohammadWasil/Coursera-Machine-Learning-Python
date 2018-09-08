# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 00:12:51 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def ComputeCentroids(X, idx, K):
    
    [m, n] = X.shape
    centroids = np.zeros((K,n))
    
    for i in range(0,K):
        centroids[i,:] = np.mean(X[(idx==(i+1)).nonzero()[0],:],axis=0) #[0], to take only first row of idx.
    
    # Above and below, both code works.
    
    #for i in range(0,K):
    #    for j in range(0,2):
    #        idxValue = (idx == (i+1)).nonzero()[0]
    #        centroids[i,j] = np.mean(X[idxValue,j])
    
    return centroids
    