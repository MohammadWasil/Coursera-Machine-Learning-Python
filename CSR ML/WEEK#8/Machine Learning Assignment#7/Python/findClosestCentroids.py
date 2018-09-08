# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 19:52:09 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def FindClosestCentroids(X,initialCentroids):
    
    K = len(initialCentroids)
    idx = np.zeros((len(X),1),dtype=np.int8)
    #print(idx.shape)
    m = X.shape[0]
    distance = np.zeros((len(initialCentroids),1))
    
    for i in range(m):
        for j in range(K):
            distance[j] = np.sum( (X[i,:] - initialCentroids[j,:])**2)
        idx[i] = np.argmin(distance)+1
    #print(idx)
    return idx  
    
  