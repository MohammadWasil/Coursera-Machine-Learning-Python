# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 23:23:48 2018

@author: Mohammad Wasil Saleem
"""

import matplotlib.pyplot as plot

def PlotData(X, y, xlim, ylim, showplot):
    
    pos = (y==1).nonzero()[0]
    ne = (y==0).nonzero()[0]   
    
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[pos, 0], X[pos, 1], marker = '+', color='black', linewidth = 1.5)
    ax.scatter(X[ne, 0], X[ne, 1], marker = 'o', color='yellow', linewidth = 1.5)
    ax.set(xlim=xlim,ylim=ylim)
    if showplot == True:
        plot.show()

    