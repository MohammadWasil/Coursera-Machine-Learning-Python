# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 18:53:35 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
import matplotlib.pyplot as plot

def PolyFeatures(X, power):
    X_poly = np.zeros(( X.shape[0], power ))
    X_poly = X
    
    for i in range(2 , power+1):
        X_poly = np.vstack(( X_poly, X**i ))
    #print(X_poly.T)        
    return X_poly.T

def FeatureNormalize( X_poly ):
    
    X_norm = X_poly
    mu = np.mean(X_poly, axis=0)
    sigma = np.std(X_poly, axis=0, ddof=1)
    X_norm = (X_poly - mu) / sigma
    
    return X_norm, mu, sigma

def PolyFit(min_x, max_x, mu, sigma, theta, power):
    x = np.arange((min_x - 15), (max_x + 25), 0.05)
    X_poly = PolyFeatures(x, power)
    X_poly = (X_poly - mu) / sigma
    X_poly = np.concatenate(( np.ones(( X_poly.shape[0], 1 )), X_poly ),axis=1)
    
    plot.plot(x, X_poly.dot(theta), '--')

# However,the polynomial fit is very complex and even drops off at the extremes.
# This is an indicator that the polynomial regression model is overfitting the training
# data and will not generalize well.        