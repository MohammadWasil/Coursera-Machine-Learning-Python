# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 18:11:03 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def gaussianKernel(x1, x2, sigma):
    summation = sum((x1-x2)**2)
    divide = summation / (2 * (sigma**2))
    
    sim = np.exp(-1 * divide)
    
    return sim