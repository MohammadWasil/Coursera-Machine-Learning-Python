# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:32:58 2018

@author: Mohammad Wasil Saleem
"""

from scipy.io import loadmat
from sklearn.svm import libsvm
import numpy as np

import plotData as pd
import visualizeBoundaryLinear as vbl
import gaussianKernel as gk
import visualizeBoundary as vb
import dataset3params as dsp

# Part 1: Loading and Visualizing Data 

print("Loading and Visualizing Dataset 1...")

data = loadmat('D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\ex6data1.mat')
X = np.require(data['X'],dtype=np.float64, requirements='C')          # 51X2     
#print(X.flags)
y = np.require(data['y'].flatten(),dtype=np.float64)          # 51X1
xlim=[0.0,4.5]
ylim=[1.5,5.0]
pd.PlotData(X, y, xlim, ylim, showplot = True)

# Part 2: Training Linear SVM 

print('Training Linear SVM ...')
C = 1.0
model = libsvm.fit(X, y, kernel='linear', tol=0.001, C=C, max_iter=20)

vbl.VisualizeBoundaryLinear(X, y, model, C)

# Part 3: Implementing Gaussian Kernel 

print('Evaluating the Gaussian Kernel ...')

x1 = np.array([1,2,1])
x2 = np.array([0,4,-1])
sigma = 2

sim = gk.gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %d, is %f:' %(sigma, sim))
print('\t\n(for sigma = 2, this value should be about 0.324652)\n')


'''
The next part will load and plot dataset 2. From
the fi
gure, you can obserse that there is no linear decision boundary that
separates the positive and negative examples for this dataset. However, by
using the Gaussian kernel with the SVM, you will be able to learn a non-linear
decision boundary that can perform reasonably well for the dataset.
'''

# Part 4: Visualizing Dataset 2 

print('Loading and Visualizing Dataset 2...')

data = loadmat('D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\ex6data2.mat')
X = np.require(data['X'],dtype=np.float64, requirements='C')          # 863X2     
#print(X.flags)
y = np.require(data['y'].flatten(),dtype=np.float64)          # 863X1
xlim=[0.0,1.0]
ylim=[0.4,1.0]

pd.PlotData(X, y, xlim, ylim, showplot = True)

# Part 5: Training SVM with RBF Kernel (Dataset 2) 

C = 1
sigma = 0.1  
gamma = 1 / (2 * ( sigma ** 2 ))
model = libsvm.fit(X, y, kernel = 'rbf', C=C, gamma=gamma)   # Train
xlim = [0.0,1.0]
ylim = [0.4,1.0]
dataSetExample = 2
vb.VisualizeBoundary(X, y, xlim, ylim, model, gamma, dataSetExample)

# Part 6: Visualizing Dataset 3 

print('Loading and Visualizing Dataset 3...')

data = loadmat('D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\ex6data3.mat')

X = np.require(data['X'],dtype=np.float64, requirements='C')                # 211X2     
#print(X.flags)
y = np.require(data['y'].flatten(),dtype=np.float64)                        # 211X1

Xval = np.require(data['Xval'],dtype=np.float64, requirements='C')          # 200X2     
yval = np.require(data['yval'].flatten(),dtype=np.float64)                  # 200X1

xlim=[-0.6,0.3]
ylim=[-0.8,0.6]

pd.PlotData(X, y, xlim, ylim, showplot = True)

# Part 7: Training SVM with RBF Kernel (Dataset 3) 

[C, sigma] = dsp.DataSet3Params(X, y, Xval, yval)
gamma = 1 / (2 * ( sigma ** 2 ))
model = libsvm.fit(X, y, kernel = 'rbf', C=C, gamma=gamma)   # Train
dataSetExample = 3
vb.VisualizeBoundary(X, y, xlim, ylim, model, gamma, dataSetExample)
