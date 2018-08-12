# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 20:47:57 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def EmailFeatures(word_indices):
    n=1899
    x=np.zeros((n,1))
    
    for i in range(len(word_indices)):
        email_indices = word_indices[i]
        x[email_indices] = 1
    
    return x