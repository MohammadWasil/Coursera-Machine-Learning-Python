# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:03:21 2018

@author: Mohammad Wasil Saleem
"""

from scipy.io import loadmat
from sklearn.svm import libsvm
import numpy as np

import processEmail as pe
import emailFeartures as ef


# Part 1: Email Preprocessing 

"""
We will be training a classi
er to classify whether a given email, x, is
spam (y = 1) or non-spam (y = 0). In particular, we need to convert each
email into a feature vector x of size n.
"""

print('Preprocessing sample email (emailSample1.txt)\n')

emailSample1 = open(r"D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\emailSample1.txt","r")
#emailSample1.close()
emailSample1 = emailSample1.read()                

#Extract Features
word_indices = pe.ProcessEmail(emailSample1)

print('\nWord indices:\n', word_indices)

# Feature Extraction 

print('\nExtracting features from sample email (emailSample1.txt)\n')

features = ef.EmailFeatures(word_indices)

print('Length of feature vector: %d' %len(features))
print('Number of non-zero entries: %d\n' %sum(features>0))

# Train Linear SVM for Spam Classification 

# spamTrain.mat contains 4000 training examples of spam
# and non-spam email, while spamTest.mat contains 1000 test examples. Each
# original email was processed using the processEmail and emailFeatures
# functions and converted into a vector x(i) wit a size of 1899.(4000X1899 for svmtrain.mat,
# 1000X1899 for svmtest.mat )

svmtrain = loadmat('D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\spamTrain.mat')
X = np.require(svmtrain['X'],dtype=np.float64, requirements='C')          # 51X2     
#print(X.flags)
y = np.require(svmtrain['y'].flatten(),dtype=np.float64)          # 51X1
                   
print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

C = 0.1
model = libsvm.fit(X, y, kernel='linear', C=C)
p = libsvm.predict(X, support=model[0], SV=model[1], nSV=model[2], sv_coef=model[3], 
            intercept=model[4],probA=model[5],probB=model[6],kernel='linear')
 
accuracyTrain = np.mean(p==y) * 100
print('Training accuracy', accuracyTrain)        

# Test Spam Classification 
svmtest = loadmat('D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\spamTest.mat')
Xtest = np.require(svmtest['Xtest'],dtype=np.float64, requirements='C')          # 51X2     
#print(X.flags)
ytest = np.require(svmtest['ytest'].flatten(),dtype=np.float64)          # 51X1

print('\nEvaluating the trained Linear SVM on a test set ...')

p = libsvm.predict(Xtest, support=model[0], SV=model[1], nSV=model[2], sv_coef=model[3], 
            intercept=model[4],probA=model[5],probB=model[6],kernel='linear')

accuracyTest = np.mean(p==ytest) * 100
print('Testing accuracy', accuracyTest)        

# Top Predictors of Spam 

supportVector = model[1]        #support vector
coefficient = model[3]        #coef

normalVector = coefficient.dot(supportVector)
normalVector = np.reshape(normalVector, -1)

indices = np.argsort(normalVector)[::-1]
vocabList = pe.GetVocabList()
vocabulary = {idx: word for word, idx in vocabList.items()}
print('\nTop predictors of spam: ')
for i in indices[:15]:
    print('%s (%f)' %(vocabulary[i], normalVector[i]))

# Try Your Own Emails 
print("\nTry Your Own Emails... \n")
emailSample1 = open(r"D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\emailSample1.txt","r").read()
emailSample2 = open(r"D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\emailSample2.txt","r").read()
spamSample1 = open(r"D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\spamSample1.txt","r").read()
spamSample2 = open(r"D:\ML\ML\CSR ML\WEEK#7\Machine Learning Assignment#6\Python\spamSample2.txt","r").read()

word_indices = pe.ProcessEmail(spamSample2)
x = ef.EmailFeatures(word_indices)

prediction = libsvm.predict(x, support=model[0], SV=model[1], nSV=model[2], sv_coef=model[3], 
            intercept=model[4],probA=model[5],probB=model[6],kernel='linear')

print('\nProcessed spamSample2')
print('Spam Classification: ', (prediction))
print('(1 indicates spam, 0 indicates not spam)\n\n')
