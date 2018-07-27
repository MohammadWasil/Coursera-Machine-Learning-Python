# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 20:03:09 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
from scipy import optimize
from scipy.io import loadmat

import displayData as dd
import nnCostFunction as nncf
import gradientChecking as gc
import predict as p

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

# load the data mat.
print('Loading and Visualizing Data ...')
data = loadmat('D:\ML\ML\CSR ML\WEEK#5\Machine Learning Assignment#4\Python\ex4data1.mat')
X = data['X']           # size 5000X400
y = data['y'].flatten() # size 5000
m = len(y)

#Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]         # Size of sel = (100, 400)

dd.displayData(sel)

weight = loadmat('D:\ML\ML\CSR ML\WEEK#5\Machine Learning Assignment#4\Python\ex4weights.mat')
Theta1 = np.zeros((25,401))
Theta2 = np.zeros((10,26))
Theta1 = weight['Theta1']       # 25X401 size, where 25=hidden layers, and 400+1 = input layers + bias
Theta2 = weight['Theta2']       # 10X26 size, where 10= output layers, and 25+1 = hidden layers + bias

Theta1 = np.reshape(Theta1, -1)
Theta2 = np.reshape(Theta2, -1)
nn_params = np.concatenate((Theta1, Theta2))

# Feedforward - Compute Cost
print('\nFeedforward Using Neural Network ...')
print('Checking Cost Function (w/o Regularization) ... ')
lambda_ = 0
cost, gradient = nncf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ )
print('Cost at parameters (loaded from ex4weights): ', cost)
print('this value should be about 0.287629')

print('\nChecking Cost Function (w/ Regularization) ... ')
lambda_ = 1
cost, gradient = nncf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ )
print('Cost at parameters (loaded from ex4weights): ', cost)
print('this value should be about 0.383770\n')

# Sigmoid Gradient
print('Evaluating sigmoid gradient...')
sigmoidGrad = np.array([-1, -0.5, 0, 0.5, 1])
sigmoidGradient = nncf.SigmoidGradient(sigmoidGrad)
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
print(sigmoidGradient)

# Initializing parameters
print('Initializing Neural Network Parameters ...')
initialTheta1 = nncf.RandInitializeWeights(input_layer_size, hidden_layer_size)
initialTheta2 = nncf.RandInitializeWeights(hidden_layer_size, num_labels)
#unrolling parameters
initialTheta1 = np.reshape(initialTheta1, -1)
initialTheta2 = np.reshape(initialTheta2, -1)
initial_nn_params = np.concatenate((initialTheta1, initialTheta2))      # unrolled parameters

# Implement Backpropagation
print('Checking Backpropagation')
#Check gradients by running checkNNGradients
lambda_ = 0
gc.checkNNGradients(lambda_)

# Once your backpropagation implementation is correct, we will now
# continue to implement the regularization with the cost and gradient.
print('Checking Backpropagation (w/ Regularization) ... ')
lambda_ = 3
gc.checkNNGradients(lambda_)

# Also output the costFunction debugging values
cost, gradient = nncf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ )
print('\nCost at (fixed) debugging parameters (w/ lambda = 3): ', cost)
print('for lambda = 3, this value should be about 0.576051')

# Training Neural Network
# We have now implemented all the code necessary to train a neural network.
print('Training Neural Network... ')
lambda_ = 3.0
# Create "short hand" for the cost function to be minimized
costFunction = lambda cf : nncf.nnCostFunction(cf,input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ )
#cost, grad = costFunction(initial_nn_params)

result = optimize.minimize( costFunction, initial_nn_params, method = 'CG', jac = True, options={ 'maxiter' : 50, 'disp' : False} )
nn_params = result.x

arrVal = hidden_layer_size * (input_layer_size + 1) 
Theta1 = nn_params[:arrVal].reshape((hidden_layer_size, input_layer_size+1 ))   
Theta2 = nn_params[arrVal:].reshape((num_labels, hidden_layer_size+1 ))

# Visualizing Weights
dd.displayData(Theta1[:, 1:])

# Implement Predict
prediction = p.Predict(Theta1, Theta2, X)
accuracy = np.mean(prediction == y) * 100

print('Training Set Accuracy: ', accuracy)





