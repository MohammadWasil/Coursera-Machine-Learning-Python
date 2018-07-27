# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 20:28:56 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np

def Sigmoid(x):
    # for large positive value, sigmoid will be close to 1.
    # for large negative value, sigmoid will be close to 0.  
    g = ( 1 / (1 + np.exp(-x)))
    
    return g

def SigmoidGradient(sigmoidGrad):
    sigGrad = Sigmoid(sigmoidGrad) * (1 - Sigmoid(sigmoidGrad))
    return sigGrad

def RandInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, L_in +1))
    
    init_epsilon = 0.12
    W = (np.random.random((L_out, L_in +1)) * (2 * init_epsilon) - init_epsilon)
    return W

def nnCostFunction(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, lambda_ ):
    [m,n] = X.shape
    X = np.concatenate((np.ones((m,1)),X), axis=1 )     # 5000X401
    
    arrVal = hidden_layer_size * (input_layer_size + 1)
    
    Theta1 = nn_params[:arrVal].reshape((hidden_layer_size, input_layer_size+1 ))   
    Theta2 = nn_params[arrVal:].reshape((num_labels, hidden_layer_size+1 ))
    Theta2_grad = np.zeros_like(Theta2)
    Theta1_grad = np.zeros_like(Theta1)
    
    # Size
    # Theta1 = 25X401
    # Theta2 = 10X26
    # X = 5000X401
    # y = 5000
    
    #numlbls = np.array([range(1, 11)])
    possible_labels = np.arange(1, num_labels + 1)
    J=0
    for i in range(m):
        # feed forward neural network.
        a1 = X[i, :]                                    # taking the pixels of the first image.401 X 1
        z2 = Theta1.dot(a1)                             # 25, 1
        a2 = Sigmoid(z2)                                # 25, 1
        a2 = np.concatenate((np.ones(1), a2), axis=0)   # 26 X 1
        z3 = Theta2.dot(a2)                             # 10X1
        a3 = Sigmoid(z3)                                # 10X1
        h = a3
        
        #yVec = (numlbls == y[i]).astype(int)
        yVec = np.vectorize(int)(possible_labels == y[i])
        J = J + sum(-yVec * np.log(h) - (1 - yVec) * np.log(1 - h))
        
        # Backpropagation.
        delta3 = h - yVec                               # 10X1 - 10X1 = 10X1
        delta2 = Theta2.T.dot(delta3) *  a2 * (1-a2)
        
        Theta2_grad += np.outer(delta3, a2)
        Theta1_grad += np.outer (delta2[1:], a1)
    
    # Scalling cost function and gradient    
    J = J/m
    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m
    
    # regularisation
    reg = (lambda_ / (2 * m)) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2) )
    
    # adding zero, sinze no regularisation in THETA[0].
    Theta1_grad += (lambda_/m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    Theta2_grad += (lambda_/m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))
    
    # unrolling
    Theta1_grad = np.reshape(Theta1_grad, -1)
    Theta2_grad = np.reshape(Theta2_grad, -1)
    gradient = np.concatenate((Theta1_grad, Theta2_grad))
    
    return (J + reg), gradient
    