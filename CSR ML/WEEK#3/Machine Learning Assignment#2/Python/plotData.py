# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 00:24:40 2018

@author: Mohammad Wasil Saleem
"""

import matplotlib.pyplot as plot
import numpy as np

def PlotData(x, y):
    # find more on 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
    pos = (y==1).nonzero()
    neg = (y==0).nonzero()
    
    plot.scatter(x[pos, 0], x[pos, 1],  color = 'black', marker = '+')
    plot.scatter(x[neg, 0], x[neg, 1], color ='yellow', marker = 'o')

def PlotDecisionBoundary(X, y, theta):
    plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
    plot_y = (-theta[0] - theta[1] * plot_x) / theta[2]
    plot.plot(plot_x, plot_y)
    plot.show()
    


