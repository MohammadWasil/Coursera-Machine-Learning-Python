# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 00:40:12 2018

@author: Mohammad Wasil Saleem
"""

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

import featureNormalize as fp
import gradientDescentMulti as gdm
import normalEqn as ne

# Getting the data and plotting it.
# x - profit
# y - population

URL = 'D:\ML\ML\CSR ML\WEEK#2\Machine Learning Assignment#1\Python\ex1data2.csv'

names = ['Size', 'NumberOfBedrooms', 'Price']

data = pd.read_csv(URL, names = names)      # 97 X 3 row by column.

size = data['Size']                
noOfVBedrooms = data['NumberOfBedrooms']
price = data['Price']

x = np.zeros((len(data),2))
x[:, 0] = size    
x[:, 1] = noOfVBedrooms                  

y = np.zeros((len(data),1))
y[:,0] = price   

m = len(y)

print('First 10 examples from the dataset: \n')
print(' x = ', x[0:10,:])
print(' y = ', y[0:10])

[X, mu, sigma] = fp.featureNormalize(x)

# increasing the shape, adding a column of ones to x
ones = np.ones((len(x),1))
X = np.hstack((ones, X))
#print(np.hstack((ones, X)))

# Gradient Descent 
# 1) Try different values of alpha
# 2) prediction (With feature normalisation)

alpha = 0.009;      #0.009, try 0.01, 0.009.
num_iters = 350;

# Init Theta and Run Gradient Descent 
theta = np.zeros((3,1))

[theta, J_History] = gdm.gradientDescentMulti(X, y, theta, alpha, num_iters)

print('Values of theta:')
print(theta)
plot.plot(J_History)
plot.title('Convergence Graph')
plot.xlabel('No Of Iterations')
plot.ylabel('Cost J')
'''
iteration = np.zeros((num_iters, 1))
for i in range(num_iters):
    iteration[i, :] = i

plot.plot(iteration, J_History)
'''
# Prediction
# Estimate the price of a 1650 sq-ft, 3 br house
# Recall that the first column of X is all-ones. Thus, it does not need to be normalized.

estimate = np.array([[1, 1650, 3]], dtype = np.float32)
estimate_norm = np.zeros((1, 3))

mu = np.mean(estimate)
sigma = np.std(estimate, ddof=1)
estimate_norm = (estimate - mu ) / sigma
estimate_norm = np.absolute(estimate_norm)

price = estimate_norm.dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house(using gradient descent)',price[0,0])

# Normal Equation
print('Solving with normal equation:')
# Again we need to load the data, since, X and y have normalised values of data(Above).
data = pd.read_csv(URL, names = names)      # 97 X 3 row by column.

size = data['Size']                
noOfVBedrooms = data['NumberOfBedrooms']
price = data['Price']

x = np.zeros((47,2))
x[:, 0] = size    
x[:, 1] = noOfVBedrooms                  

y = np.zeros((47,1))
y[:,0] = price

theta = ne.NormalEquation(X, y)
print('Values of theta:')
print(theta)

estimate = np.array([[1, 1650, 3]], dtype = np.float32)
price = estimate_norm.dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house(using normal equations)', price[0,0])
