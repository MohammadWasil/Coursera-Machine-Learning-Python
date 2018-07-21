# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:20:31 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
from scipy.io import loadmat

import displayData as dd

def Sigmoid(x):
    # for large positive value, sigmoid will be close to 1.
    # for large negative value, sigmoid will be close to 0.  
    g = ( 1 / (1 + np.exp(-x)))    
    return g

def Predict(theta1, theta2, X):
    [m,n] = X.shape
    a1 = np.hstack((np.ones((m,1)), X))
    z2 = Sigmoid(a1.dot(theta1.T))
    a2 = np.hstack((np.ones((m,1)), z2))
    z3 = Sigmoid(a2.dot(theta2.T))
    prediction = 1 + np.argmax(z3, axis=1)
    return prediction


# Since the images are of size 20X20, this gives us 400 input layer
# units (excluding the extra bias unit which always outputs +1).
# You have been provided with a set of network parameters (theta(1);theta(2)) already trained.
# The parameters have dimensions that are sized for a neural network with 25 units 
# in the second layer and 10 output units (corresponding to the 10 digit classes).

# load the data mat.
print('Loading and Visualizing Data ...')
data = loadmat('D:\ML\ML\CSR ML\WEEK#4\Machine Learning Assignment#3\Python\ex3data1.mat')
X = data['X']           # size 5000X400
y = data['y'].flatten() # size 5000
m = len(y)

#Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]         # Size of sel = (100, 400)

dd.displayData(sel)

weight = loadmat('D:\ML\ML\CSR ML\WEEK#4\Machine Learning Assignment#3\Python\ex3weights.mat')
theta1 = np.zeros((25,401))
theta2 = np.zeros((10,26))
theta1 = weight['Theta1']       # 25X401 size, where 25=hidden layers, and 400+1 = input layers + bias
theta2 = weight['Theta2']       # 10X26 size, where 10= output layers, and 25+1 = hidden layers + bias

num_labels = len(theta2)

# You need to return the following variables correctly 
p = np.zeros((len(X), 1));
predict = Predict(theta1, theta2, X)
accuracy = np.mean(predict == y) * 100
print('Training set accuracy', accuracy)

rp = np.random.permutation(m)

for i in range(m):
    inputs = input('Paused - press enter to continue, q to exit: ')
    if inputs == 'q':
        break
    else :#if inputs == 'w':
        test = X[rp[i], :].reshape(1,-1)
        print(test.shape)
        dd.displayData(test)
        pred = Predict(theta1, theta2, test)
        print('Neural Network Prediction: ',  pred[0] )
        print('Digit: ', (pred[0] % 10) )
        







