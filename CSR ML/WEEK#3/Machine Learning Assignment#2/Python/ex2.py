# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 00:09:18 2018

@author: Mohammad Wasil Saleem
"""
import plotData as pld
import costFunction as cf

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plot


URL = 'D:\ML\ML\CSR ML\WEEK#3\Machine Learning Assignment#2\Python\ex2data1.csv'

names = ['Exam 1 Score', 'Exam 2 Score', 'Decision']

data = pd.read_csv(URL, names = names)

x = np.zeros((len(data), 2))
x[:, 0] = data['Exam 1 Score']
x[:, 1] = data['Exam 2 Score']

y = np.zeros((len(data), 1))
y[:, 0] = data['Decision']

# Plotting the data
fig = plot.figure()
ax = fig.add_subplot(111)
    
ax.set(xlim=[30,100], ylim=[30,100])
plot.title('Scatter Plot of Training Data')
plot.xlabel('Score 1 Exam')
plot.ylabel('Score 2 Exam')

pld.PlotData(x, y)

### Cost Function and Gradient Descent.

# increasing the shape, adding a column of ones to x
ones = np.ones((len(data),1))
X = np.hstack((ones,x))

# Initialize fitting parameters
theta = np.zeros(3)
J, grad = cf.costFunction(theta, X, y)

print('Cost at initial theta (zeros): ', J[0]);
print('Expected cost (approx): 0.693');
print('Gradient at initial theta (zeros): ');
print(grad);
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

# Testong theta at different values.
theta = np.array([-24, 0.2, 0.2])
J, grad = cf.costFunction(theta, X, y)

print('\nCost at test theta: ', J[0]);
print('Expected cost (approx): 0.218');
print('Gradient at test theta: ');
print(grad);
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

#  Optimizing (as fminunc in octave)
# https://docs.scipy.org/doc/scipy-0.10.0/reference/tutorial/optimize.html
# https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.minimize.html
wrapping =lambda t : cf.costFunction(t, X, y)[0]
result = minimize(wrapping, theta, method='Nelder-Mead', options= {'maxiter': 400, 'disp': False})
theta = result.x
cost = result.fun

print('Cost at theta found by Scipy, using optimize: ', cost);
print('Expected cost (approx): 0.203');
print('theta: ');
print(theta);
print('Expected theta (approx): ');
print(' -25.161\n 0.206\n 0.201\n');

 # plot the decision boundary
pld.PlotDecisionBoundary(X[:, 1:], y, theta)

# Predict
estimate_score = np.array([1, 45, 85])
prob = cf.Sigmoid(estimate_score.dot(theta));

print('For a student with scores 45 and 85, we predict an admission probability of ', prob);
print('Expected value: 0.775 +/- 0.002\n\n');

# Accuracies.
predictions = cf.Predict(X, theta)
# To find out how much predictions is eualy to y
accuracy = np.mean(predictions == y[:, 0]) * 100

print('Train Accuracy: ', accuracy);
print('Expected accuracy (approx): 89.0');