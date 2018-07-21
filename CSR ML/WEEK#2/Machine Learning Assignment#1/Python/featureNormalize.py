# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 01:28:22 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def featureNormalize(x):
    X_norm = x
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    X_norm = (x - mu ) / sigma
    
    return X_norm, mu, sigma