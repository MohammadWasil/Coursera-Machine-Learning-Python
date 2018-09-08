# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:01:18 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def ProjectData(X_norm, U, K):
    m = X_norm.shape[0]
    Z = np.zeros((m, 1))
    
    U_reduced = U[:, 0:K]
    Z = X_norm.dot(U_reduced)
    
    return Z