# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:36:26 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def mapFeature(X1, X2):
    degree = 6
    '''
    out = np.ones((len(X1), 1))
   
    for i in range(degree+1):
       for j in range(i+1):
          out = np.append(out, (X1 ** (i - j)) * (X2 ** j))
    '''       
    m = X1.shape[0] if X1.shape else 1
    cols = [np.ones(m)]
    for i in range(1, degree + 1):
        for j in range(i + 1):
            cols.append((X1 ** (i - j)) * (X2 ** j))
            
    return np.vstack(cols).T

    