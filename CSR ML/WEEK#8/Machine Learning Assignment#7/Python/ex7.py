# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 19:46:22 2018

@author: Mohammad Wasil Saleem
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plot

import findClosestCentroids as fcc
import computeCentroids as cc
import runKMeans as rkm
import kMeansInitCentroids as kmic

# Find Closest Centroids 

# To help you implement K-Means, we have divided the learning algorithm 
# into two functions -- findClosestCentroids and computeCentroids. In this
# part, you should complete the code in the findClosestCentroids function. 

print('Finding closest centroids.\n')

data2 = loadmat('D:\ML\ML\CSR ML\WEEK#8\Machine Learning Assignment#7\Python\ex7data2.mat')
X = data2['X']
#print(X.shape)
K = 3
initialCentroids = np.array([[3,3],[6,2],[8,5]])
idx = fcc.FindClosestCentroids(X,initialCentroids)

print('Closest centroids for the first 3 examples: ')
print(idx[0:3])
print('\n(the closest centroids should be 1, 3, 2 respectively)\n')

# Compute Means

print('Computing centroids means.')

centroids = cc.ComputeCentroids(X, idx, K)
print('Centroids computed after initial finding of closest centroids: \n')
print(centroids)

print('\n(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

# K-Means Clustering 

print('\nRunning K-Means clustering on example dataset.\n')
max_iters = 10
#[centroids, idx] = rkm.RunKMeans(X, initialCentroids, max_iters, True)
print('\nK-Means Done.\n')

# K-Means Clustering on Pixels
print('\nRunning K-Means clustering on pixels from an image.\n')

#im=plot.imread('bird_small.png')
#print(im.shape)

mat=loadmat('bird_small.mat')
A=mat['A']

A=A
img_size = A.shape
print(img_size)

X=A.reshape(img_size[0]*img_size[1],3,order='F').copy()
print(X.shape)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly. 
# You should complete the code in kMeansInitCentroids.m before proceeding

initialCentroids = kmic.KMeansInitCentroids(X,K)

[centroids, idx] = rkm.RunKMeans(X, initialCentroids, max_iters,False)
print(centroids.shape)
print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = fcc.FindClosestCentroids(X, centroids)
#centroids=centroids[-1]

print(idx.shape)                          #16384       1

X_recovered=np.zeros((idx.shape[0],3))
#print(idx.shape)
#print(centroids.shape)
#for i in range(X_recovered.shape[0]):
#    X_recovered[i] = centroids[idx[i]]

X_recovered = centroids[idx.astype(int),:]
print(X_recovered.shape)                  #16384  1   3
X_recovered = X_recovered.reshape(img_size[0],img_size[1],3,order='F')
print(X_recovered.shape)                  #16384       3

#fig, (ax1,ax2) = plot.subplots(1,2,figsize=(13,9))
fig1 = plot.figure()
ax1 = fig1.add_subplot(111)
ax1.imshow(A)

fig2 = plot.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(X_recovered)







