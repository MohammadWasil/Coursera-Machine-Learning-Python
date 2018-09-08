# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 01:11:45 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def KMeansInitCentroids(X,K):
    randidx = np.random.permutation(len(X))
    centroids = X[randidx[0:K],:]
    return centroids