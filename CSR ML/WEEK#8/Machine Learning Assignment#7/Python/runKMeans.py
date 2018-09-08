# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 01:11:30 2018

@author: Mohammad Wasil Saleem
"""
import numpy as np

import findClosestCentroids as fcc
import computeCentroids as cc
import plotProgress as pp

def RunKMeans(X, initial_centroids, max_iters, showgraph):
    [m, n] = X.shape
    K = len(initial_centroids)
    centroids = initial_centroids
    previous_centroids = centroids
    
    #centroidsArray = []
    #previousArray = []
    
    idx = np.zeros((m, 1))
    
    for i in range(0,max_iters):
        idx = fcc.FindClosestCentroids(X,centroids )
        
        if showgraph==True:
            inputs = input('Paused - press enter to continue, q to exit: ')
            if inputs == 'q':
                break
            else :
                pp.PlotCentroid(X, centroids, previous_centroids,idx, K, i)
                previous_centroids = centroids
                
                centroids = cc.ComputeCentroids(X,idx,K)
                
        elif showgraph==False:
            previous_centroids = centroids
            centroids = cc.ComputeCentroids(X,idx,K)
            
            
            
    return centroids,idx