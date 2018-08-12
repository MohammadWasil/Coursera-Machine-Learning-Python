# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:49:03 2018

@author: Mohammad Wasil Saleem
"""

import plotData as pd

import matplotlib.pyplot as plot
import numpy as np

def VisualizeBoundaryLinear(X, y, model, C):
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.libsvm.fit.html
    supportVector = model[1]        #support vector
    coefficient = model[3]        #coef
    intercept = model[4]        #intercept
    
    normalVector = coefficient.dot(supportVector)
    normalVector = np.reshape(normalVector, -1)
    
    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = -(normalVector[0] * xp + intercept) / normalVector[1]
    xlim=[0.0,4.5]
    ylim=[1.5,5.0]
    pd.PlotData(X, y, xlim, ylim, showplot = False)
    plot.plot(xp,yp)
    plot.title('SVM Decision Boundary with C = %f (Example Dataset 1)' % C)
    plot.show()
