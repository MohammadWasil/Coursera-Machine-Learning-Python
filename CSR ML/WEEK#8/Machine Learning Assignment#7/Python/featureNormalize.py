# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:08:14 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def FeatureNormalize(x):
    X_norm = x
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    X_norm = (x - mu ) / sigma
    
    return X_norm, mu, sigma