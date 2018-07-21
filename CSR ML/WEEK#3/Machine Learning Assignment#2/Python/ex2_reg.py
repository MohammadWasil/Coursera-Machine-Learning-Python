# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:13:30 2018

@author: Mohammad Wasil Saleem
"""

import plotData as pld
import mapFeature as mf
import costFunctionReg as cfr


import pandas as pd
import matplotlib.pyplot as plot
from scipy.optimize import minimize
import numpy as np

URL = 'D:\ML\ML\CSR ML\WEEK#3\Machine Learning Assignment#2\Python\ex2data2.csv'

names = ['Microchip Test 1', 'Microchip Test 2', 'Decision']

data = pd.read_csv(URL, names = names)

X = np.zeros((len(data), 2))
X_ori = np.zeros((len(data), 2))
X_ori[:, 0] = X[:, 0] = data['Microchip Test 1']
X_ori[:, 1] = X[:, 1] = data['Microchip Test 2']

y = np.zeros((len(data), 1))
y[:, 0] = data['Decision']

# Plotting the data
fig = plot.figure()
ax = fig.add_subplot(111)
    
plot.title('Scatter Plot of Training Data')
plot.xlabel('Microchip Test 1')
plot.ylabel('Microchip Test 2')

pld.PlotData(X, y)

# Feature Mapping.
X = mf.mapFeature(X[:, 0], X[:, 1])
m, n = X.shape
theta = np.zeros(n)


lambdaa = 1

J, grad = cfr.costFunctionReg(theta, X, y, lambdaa)

print('Cost at initial theta (zeros): ', J[0])
print('Expected cost (approx): 0.693')

print('Gradient at initial theta (zeros) - first five values only: ', grad[:,0:5]);
print('Expected gradients (approx) - first five values only: ');
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

###
# Tryinmg different values of lambda (0, 1, 10, 100).
###

# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html
wrapping =lambda t : cfr.costFunctionReg(t, X, y, lambdaa)[0]
result = minimize( wrapping, theta, method='CG', options={ 'maxiter': 400, 'disp': False})
theta = result.x

p = cfr.Predict(X, theta)

accuracy = np.mean(p == y[:, 0]) * 100
print('Train Accuracy: ', accuracy);
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n');