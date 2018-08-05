# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:00:14 2018

@author: Mohammad Wasil Saleem
"""
import matplotlib.pyplot as plot
from scipy.io import loadmat
import numpy as np

import linearRegCostFunction as lrcf
import trainLinearRegression as tlr
import learningCurve as lc
import polyFeatures as pf
import validationCurve as vc

# Loading and Visualizing Data 

# This dataset is divided into three parts:
# • A training set that your model will learn on: X, y
# • A cross validation set for determining the regularization parameter: Xval, yval
# • A test set for evaluating performance. These are \unseen" examples which your model 
# did not see during training: Xtest, ytest


print('Loading and Visualizing Data ...')
data = loadmat('D:\ML\ML\CSR ML\WEEK#6\Machine Learning Assignment#5\Python\ex5data1.mat')
X = data['X']       # 12X1
y = data['y']       # 12X1
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

y = np.reshape(y, -1)       
yval = np.reshape(yval, -1)       

m = len(y)

fig = plot.figure()
ax = fig.add_subplot(111)
ax.scatter(X, y, marker = 'x', color='red', linewidth = 1.5)
ax.set_xlim(-50, 40)
plot.title('Data Visualization')
plot.xlabel('Change in water level (x)')
plot.ylabel('Water flowing out of the dam (y)')
plot.show()

# Regularized Linear Regression Cost 
print('\nCalculating Cost Function for Regularized Linear Regression...')

ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))        # 12 X 2

onesVal = np.ones((Xval.shape[0], 1))
Xval = np.hstack((onesVal, Xval))        # 12 X 2

theta = np.array([1,1])         # 2, 1
J, grad = lrcf.linearRegCostFunction(X, y, theta, 1)
print('Cost at theta = [1 ; 1]: ', J)
print('(this value should be about 303.993192)')


# Regularized Linear Regression Gradient 

print('\nCalculating Gradient for Regularized Linear Regression...')
J, grad = lrcf.linearRegCostFunction(X, y, theta, 1)
print('Gradient at theta = [1 ; 1]: ', grad)
print('(this value should be about [-15.303016; 598.250744])')

# Train Linear Regression
lambda_ = 0
theta = tlr.TrainLinearRegression(X, y,  lambda_)

fig1 = plot.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(X[:, 1], y, marker = 'x', color='red', linewidth = 1.5)
ax1.plot(X[:, 1],  X.dot(theta), '--')       # To not include the first column of X
ax1.set_xlim(-50, 40)
plot.title('Linear Fit')
plot.xlabel('Change in water level (x)')
plot.ylabel('Water flowing out of the dam (y)')
plot.show()
# While visualizing the best fitt as shown is one possible way to debug your learning
# algorithm, it is not always easy to visualize the data and model. In the next
# section, we will implement a function to generate learning curves that can
# help to debug learning algorithm even if it is not easy to visualize the data.

# Learning Curve for Linear Regression
print('\nPlottiong Learning curve for Linear Regression')
lambda_ = 0
error_train, error_cross_validation = lc.LearningCurve(X, y, Xval, yval, lambda_)

print('Training Example\tTrain Error\tCross Validation Error')
for i in range(1, m+1):
    print('\t%d\t\t%f\t%f' %(i,error_train[i],error_cross_validation[i]))

fig2 = plot.figure()
ax2 = fig2.add_subplot(111)
plot.xlabel('Number of training Examples')
plot.ylabel('Error')
plot.title('Linear Regression Learning Curve')
ax2.plot(range(m), error_train[1:, :])
ax2.plot(range(m), error_cross_validation[1:, :])
plot.show()

# Feature Mapping for Polynomial Regression 

print('Feature Mapping for Polynomial Regression...')
power = 8

# Map X onto Polynomial Features and Normalize

# The function should return a mXp matrix X_poly,
# where column 1 holds the original values of X, column 2 holds the values of
# X.^2, column 3 holds the values of X.^3, and so on. Note that we don't
# have to account for the zero-eth power in this function

X_poly = pf.PolyFeatures(X[:,1], power)
# It turns out that if we run the training directly on the projected data, will
# not work well as the features would be badly scaled (e.g., an example with
# x = 40 will now have a feature x[,8] = 40^8 = 6.5 X 10^12). Therefore, we will
# need to use feature normalization.
X_poly, mu, sigma = pf.FeatureNormalize(X_poly)
X_poly = np.concatenate(( np.ones(( X_poly.shape[0],1 )),X_poly ),axis=1)

#Map X_poly_test and normalize (using mu and sigma)
#X_poly_test = pf.PolyFeatures(Xtest, power)
#X_poly_test = (X_poly_test - mu) / sigma
#X_poly_test = np.concatenate(( np.ones(( X_poly_test.shape[0], 1 )), X_poly_test ))

X_poly_val = pf.PolyFeatures(Xval[:, 1], power)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.concatenate(( np.ones(( X_poly_val.shape[0], 1 )), X_poly_val ),axis=1)

print('Normalized Training Example 1:');
print(X_poly[0,:])

# Learning Curve for Polynomial Regression 
lambda_ = 0
theta = tlr.TrainLinearRegression(X_poly,y,lambda_)

# Plot training data and fit
fig3 = plot.figure()
ax3 = fig3.add_subplot(111)
ax3.scatter(X[:,1], y, marker = 'x', color='red', linewidth = 1.5)
pf.PolyFit(X.min(), X.max(), mu, sigma, theta, power)
plot.title('Polynomial Reression Fit(Lambda = 0.00)')
plot.xlabel('Change in water level (x)')
plot.ylabel('Water flowing out of the dam (y)')
plot.show()

error_train, error_cross_validation = lc.LearningCurve(X_poly, y, X_poly_val, yval, lambda_)

fig4 = plot.figure()
ax4 = fig4.add_subplot(111)
plot.xlabel('Number of training Examples')
plot.ylabel('Error')
plot.title('Polynomial Reression Learning Curve')
ax4.plot(range(m), error_train[1:, :])
ax4.plot(range(m), error_cross_validation[1:, :])
plot.show()
# Low training error is low, but the cross validation error is high. 
# There is a gap between the training and cross validation errors, indicating a high
# variance problem.

print('Polynomial Regression lambda = %f' %(lambda_))
print('Training Example\tTrain Error\tCross Validation Error')
for i in range(1, m+1):
    print('\t%d\t\t%f\t%f' %(i,error_train[i],error_cross_validation[i]))

# one way to combat te overfittin(hih-variance) problem is to add reularization to the model

# Validation for Selecting Lambda 
lambda_vec, error_train, error_cross_validation = vc.ValidationCurve(X_poly, y, X_poly_val, yval)

fig5 = plot.figure()
ax5 = fig5.add_subplot(111)
plot.xlabel('Lambda')
plot.ylabel('Error')
ax5.plot(lambda_vec, error_train)
ax5.plot(lambda_vec, error_cross_validation)
plot.show()

print('lambda\t\tTrain Error\tCross Validation Error')
for i, lambda_ in enumerate(lambda_vec,0):
    print('%f\t%f\t%f' %(lambda_,error_train[i],error_cross_validation[i]))

