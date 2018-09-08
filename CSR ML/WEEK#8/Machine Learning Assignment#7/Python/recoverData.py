# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:14:07 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def RecoverData(Z, U ,K):
    
    X_rec = np.zeros((Z.shape[0], U.shape[0] ))
    
    U_reduced = U[:,0:K]
    X_rec = Z.dot(U_reduced.T)
    
    return X_rec  