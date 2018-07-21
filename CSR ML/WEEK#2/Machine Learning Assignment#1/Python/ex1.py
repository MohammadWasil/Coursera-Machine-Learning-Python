# -*- coding: utf-8 -*-
"""

Machine Learning Assignment#1

Created on Tue Jul 10 19:40:47 2018

@author: Mohammad Wasil Saleem


"""

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np


# Other Python files.
import computeCost as cp
import gradientDescent as gd

# Getting the data and plotting it.
# x - profit
# y - population

URL = 'D:\ML\ML\CSR ML\WEEK#2\Machine Learning Assignment#1\Python\ex1data1.csv'

names = ['Profit', 'Population']

data = pd.read_csv(URL, names = names)      # 97 X 2 row by column.

profit = data['Profit']                
population = data['Population']

x = np.zeros((97,1))
x = profit                                    


y = np.zeros((97,1))
y = population   


fig = plot.figure()
ax = fig.add_subplot(111)
ax.set_xlim(4, 22.5)

plot.title('Plot Data')
plot.xlabel('Profit')
plot.ylabel('Population')
plot.scatter(data['Profit'], data['Population'], color = (0.5, 0.1, 0.5, 0.6), marker = '+')

### Plotting the data -End

### Cost Function and Gradient Descent.

# increasing the shape, adding a column of ones to x
prof = np.zeros((97,1))
ones = np.ones((97,1))
X = np.hstack((ones,prof))
X[:, 1] = profit


theta = np.zeros(2)  #initialisng fitting parameter theta

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

# compute and display initial cost
J = cp.computeCost(X, y, theta);
print('With theta = [0 ; 0]\nCost computed =\n', J)
print('Expected cost value (approx) 32.07\n')

# compute cost at some other value of theta,

#theta = np.array([-1, 2])
#J = cp.computeCost(X, y, theta);
#print('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J)
#print('Expected cost value (approx) 54.24\n')

# Gradient Descent

# Now to calculate theta value, we'll apply gradient descent.

theta = gd.gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:\n')
print('\n', theta)
print(theta.shape)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

### Cost function and gradient descent End.

## plot the linear fit, on the above plot.

ax.plot(X[:, 1], X.dot(theta), color='red')

plot.show()

## end- plot the linear fit.

### Predict values for population sizes of 35,000 and 70,000

inputX = np.array([1, 3.5] ) 
predict1 = inputX.dot(theta);
print('For population = 35,000, we predict a profit of ',   predict1*10000);

inputX = np.array([1, 7.0] ) 
predict2 = inputX.dot(theta);
print('For population = 70,000, we predict a profit of ',   predict2*10000);

### End- Predict values for population sizes of 35,000 and 70,000

### Contour plots.

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);

# initialize J_vals to a matrix of 0's
j_vals = np.zeros((len(theta0_vals), len(theta1_vals)));

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [theta0_vals[i], theta1_vals[j] ]
        j_vals[i, j] = cp.computeCost(X, y, t)

plot.title('Contour Plotting')
plot.xlabel('theta0')
plot.ylabel('theta1')
plot.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20))
### End- plot contour

### 3D graph between, J(theta0, theta1), theta0, theta1
fig2 = plot.figure()
ax2 = fig2.add_subplot(111, projection = '3d')
theta0_temp, theta1_temp = np.meshgrid(theta0_vals, theta1_vals)
plot.title('3-D Graph')
plot.xlabel('theta0')
plot.ylabel('theta1')
ax2.plot_surface(theta0_temp, theta0_temp, j_vals)
plot.show()




