# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 23:40:09 2018

@author: Mohammad Wasil Saleem
"""

from scipy.io import loadmat
import matplotlib.pyplot as plot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import featureNormalize as fn
import pca as pca
import drawLine as dl
import projectData as pd
import recoverData as rd
import displayData as dd

import runKMeans as rkm
import kMeansInitCentroids as kmic


# Load Example Dataset

print('Visualizing example dataset for PCA.\n')

data = loadmat('ex7data1.mat')

X = data['X']
#print(X.shape)

fig = plot.figure()
ax = fig.add_subplot(111)

ax.scatter(X[:,0],X[:,1],marker="o",facecolors='none', edgecolors='blue',linewidth=1.0)
ax.set(xlim = [0.5,6.5], ylim = [2,8])
plot.title('Example Dataset 1')
plot.show()
# Principal Component Analysis

print('\nRunning PCA on example dataset.\n')

[X_norm, mu, sigma] = fn.FeatureNormalize(X)

[U, S] = pca.PCA(X_norm)

fig2 = plot.figure()
ax2 = fig2.add_subplot(111)

ax2.hold(True)

ax2.scatter(X[:,0],X[:,1],marker="o",facecolors='none', edgecolors='blue',linewidth=0.5)
ax2.set(xlim = [0.5,6.5], ylim = [2,8])
plot.title('Computed eigenvectors of the dataset')

dl.DrawLine(mu, mu + 1.5 * S[0,0] * U[:,0].T,ax2, c = 'k', LineWidth = 2)
dl.DrawLine(mu, mu + 1.5 * S[1,1] * U[:,1].T,ax2, c = 'k', LineWidth = 2)
ax2.hold(False)

plot.show()


print('Top eigenvector: ')
print(' U(:,1) = {:f}  {:f}'.format(U[0,0], U[1,0]))
print('\n(you should expect to see -0.707107 -0.707107)\n')

# Dimension Reduction

print('\nDimension reduction on example dataset.\n')

fig3 = plot.figure()
ax3 = fig3.add_subplot(111)

ax3.scatter(X_norm[:,0],X_norm[:,1],facecolors='none', edgecolors='blue',linewidth=0.5)
ax3.set(xlim = [-4,3], ylim = [-4,3])
plot.show()

#  Project the data onto K = 1 dimension
K = 1
Z = pd.ProjectData(X_norm, U, K)
print('Projection of the first example: %f\n', Z[0])
print('\n(this value should be about 1.481274)\n')


X_rec  = rd.RecoverData(Z, U, K);
print('Approximation of the first example: %f %f\n', X_rec[0, 0], X_rec[0, 1])
print('\n(this value should be about  -1.047419 -1.047419)\n')

#  Draw lines connecting the projected points to the original points
fig4 = plot.figure()
ax4 = fig4.add_subplot(111)

ax4.hold(True)

ax4.scatter(X_norm[:,0],X_norm[:,1],facecolors='none', edgecolors='blue',linewidth=0.5)
ax4.scatter(X_rec[:,0],X_rec[:,1],facecolors='none', edgecolors='red',linewidth=0.5)
ax4.set(xlim = [-4,3], ylim = [-4,3])

for i in range(len(X_norm)):
    dl.DrawLine(X_norm[i,:], X_rec[i,:],ax4, c='k', lineWidth= 1)       #linestyle='--'

ax4.hold(False)
plot.show()

# Loading and Visualizing Face Data

# Each row of X corresponds to one face image (a row vector of length 1024),each image 32 X 32 in grayscale.
# 32*32=1024

print('\nLoading face dataset.\n')

dataFace = loadmat('ex7faces.mat')
X = np.array(dataFace["X"])
#  Display the first 100 faces in the dataset
dd.DisplayData(X[0:100, :])

# PCA on Face Data: Eigenfaces

print('Running PCA on face dataset.(this might take a minute or two ...)\n')
examples = 36
print('Showing first %d examples' %(examples))

[X_norm, mu, sigma] = fn.FeatureNormalize(X)

[U, S] = pca.PCA(X_norm)

dd.DisplayData(U[:, 0:examples].T)

# Dimension Reduction for Faces 

print('Dimension reduction for face dataset.\n')

'''
This allows you to use your learning algorithm with a smaller input size (e.g., 100 dimensions)
instead of the original 1024 dimensions. This can help speed up your learning algorithm.
'''

K = 100;
Z = pd.ProjectData(X_norm, U, K);

print('The projected data Z has a size of: ')
print('{:d}  {:d}'.format(Z.shape[0],Z.shape[1]))

# Visualization of Faces after PCA Dimension Reduction(Recovering)

print('Visualizing the projected (reduced dimension) faces.\n')

K = 100
X_rec  = rd.RecoverData(Z, U, K)

# Display normalized data
print('\nDisplaying normalized data')
dd.DisplayData(X_norm[:100,:])

# Display reconstructed data from only k eigenfaces
print('\nDisplaying reconstructed data from only k eigenfaces')
dd.DisplayData(X_rec[:100,:])

# Optional (ungraded) Exercise: PCA for Visualization 
# From ex7.py file.
mat=loadmat('bird_small.mat')
A=mat['A']

A=A/255
img_size = A.shape
print(img_size)

X=A.reshape(img_size[0]*img_size[1],3,order='F').copy()
print(X.shape)

K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly. 
# You should complete the code in kMeansInitCentroids.m before proceeding

initialCentroids = kmic.KMeansInitCentroids(X,K)

[centroids, idx] = rkm.RunKMeans(X, initialCentroids, max_iters,False)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int).flatten()

#  Setup Color Palette
palette = dd.hsv(K)
colors = np.array([palette[int(i)] for i in idx[sel]])

fig5 = plot.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(X[sel,0],X[sel,1],X[sel,2],marker="o",facecolors='none', edgecolors=colors,linewidth=1)
plot.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

# Optional (ungraded) Exercise: PCA for Visualization

[X_norm, mu, sigma] = fn.FeatureNormalize(X)

[U, S] = pca.PCA(X_norm)

Z = pd.ProjectData(X_norm, U, 2)
#print(Z.shape)

# Plot in 2D
fig6 = plot.figure()
ax6 = fig6.add_subplot(111)

ax6.scatter(Z[sel, 0], Z[sel, 1], marker="o",facecolors='none', edgecolors='blue')

plot.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
plot.show()
