# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 01:28:15 2018

@author: Mohammad Wasil Saleem
"""

import matplotlib.pyplot as plot
import numpy as np
"""
def Lineistory(centroids,previous,ax):
    
    for i in range(0,3):
        for j in range(0,2):
            ax.plot(centroids[i][j], previous[i][j], 'k-', lw=1)
            #print(previousArray[i][j])
"""
def PlotData(X, idx, K, ax):
   
    jet=plot.get_cmap('jet')
    colors=iter(jet(np.linspace(0,500,1500)))
    
    for i in range(0,K):
        idxValue = (idx==(i+1)).nonzero()[0]
        ax.scatter(X[idxValue,0],X[idxValue,1],marker="o",color=next(colors),linewidth=0.05)
    
    """
    idxValue2 = (idx==(2)).nonzero()[0]
    plot.scatter(X[idxValue2,0],X[idxValue2,1],marker="o",color=green,linewidth=0.05)
    
    idxValue3 = (idx==(3)).nonzero()[0]
    plot.scatter(X[idxValue3,0],X[idxValue3,1],marker="o",color=blue,linewidth=0.05)
    """

def PlotCentroid(X, centroids, previous, idx, K, i):
    #print(centroids[0][0])
    #print(previous[0])
    #centroidsArray.append(centroids)
    #previousArray.append(previous)        
    #print(centroidsArray[0][1])     
    #print(len(previousArray))
    
    fig = plot.figure()
    ax = fig.add_subplot(111)
    PlotData(X, idx, K, ax)
    
    ax.scatter(centroids[:,0],centroids[:,1],marker="x",color="black",linewidth=2.0)
    plot.title('iteration number %d' %(i+1))
    #Lineistory(centroids,previous,ax)
    plot.show()