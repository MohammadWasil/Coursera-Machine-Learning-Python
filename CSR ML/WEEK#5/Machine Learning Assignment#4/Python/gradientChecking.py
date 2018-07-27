# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 01:22:16 2018

@author: Mohammad Wasil Saleem
"""
import numpy as np
from numpy.linalg import norm

import nnCostFunction as nncf

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.sin(1 + np.arange((1 + fan_in) * fan_out)).reshape((1 + fan_in, fan_out)).T 
    return (W/10)
    
def computeNumericalGradient(J, theta):
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    e = 1e-4;
    for i in range(0, len(theta)):
        # Set perturbation vector
        perturb[i] = e
        loss1, grad = J(theta - perturb)
        loss2, grad = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[i] = (loss2 - loss1) / (2*e)
        perturb[i] = 0
        
    return numgrad


def checkNNGradients(lambda_):
    #    Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using computeNumericalGradient). These two gradient computations should
    #   result in very similar values.
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;
    
    # generating some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
    X  = debugInitializeWeights(m, input_layer_size - 1);
    y = 1 + np.arange(1, m+1) % num_labels
    
    Theta1 = np.reshape(Theta1, -1)
    Theta2 = np.reshape(Theta2, -1)
    nn_params = np.concatenate((Theta1, Theta2))

    costFunction = lambda cf : nncf.nnCostFunction(cf,input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ )
    cost, grad = costFunction(nn_params)
    numgrad = computeNumericalGradient(costFunction, nn_params)
    compareGradients = np.vstack((grad, numgrad))
    print(compareGradients.T)
    print('The above two columns you get should be very similar.\nLeft-Your Numerical Gradient, Right-Analytical Gradient')

    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = norm(numgrad-grad)/norm(numgrad+grad)
    
    print('If your backpropagation implementation is correct, then \n the relative difference will be small (less than 1e-9). \n\nRelative Difference: ', diff)
             
