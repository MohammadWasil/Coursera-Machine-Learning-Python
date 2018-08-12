# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 19:36:15 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
from sklearn.svm import libsvm
import matplotlib.pyplot as plot

import plotData as pd

def VisualizeBoundary(X, y, xlim, ylim, model, gamma, dataSetExample):
    
    pd.PlotData(X, y, xlim, ylim, showplot = False)
    
    x1plot = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 100)
    x2plot = np.linspace(np.amin(X[:, 1]), np.amax(X[:, 1]), 100)

    X1, X2 = np.meshgrid(x1plot, x2plot)
    [m,n] = X1.shape
    
    vals = np.zeros_like((X1))
    for i in range(m):
        this_X = np.require(np.vstack((X1[:,i], X2[:,i])).T, requirements='C')

        #http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.fit.html
        #http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.predict.html
        vals[:,i] = libsvm.predict(this_X, support=model[0], SV=model[1], nSV=model[2], sv_coef=model[3], 
            intercept=model[4],probA=model[5],probB=model[6],kernel='rbf', gamma=gamma)
           
    plot.contour(X1, X2, vals, colors = 'red')
    plot.title('SVM (Gaussian Kernel) Decision Boundary (Example Dataset %d)' %(dataSetExample))
    plot.show()
    
    
    