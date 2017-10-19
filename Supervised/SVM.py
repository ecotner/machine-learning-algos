# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:45:54 2017

Algorithm for implementing a Support Vector Machine (SVM). The process is
essentially to minimize the 'hinge loss' function on each class (with
regularization if applicable.

The classifier computes for each example the classification vector z=Wx+b (where
x is the input feature vector, W is a matrix of weights, and b is the bias), and
minimizes the 'hinge loss'

@author: Eric Cotner
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the kernel transformation into new features
def K_RBF(X_train, X_test, sigma):
    m_train = X_train.shape[1]
    m_test = X_test.shape[1]
    F = np.zeros((m_train,m_test))
    for i in range(m_train):
        for j in range(m_test):
            F[i,j] = np.exp(-np.linalg.norm(X_train[:,i] - X_test[:,j])**2/(2*sigma**2))
    return F

def K_linear(X):
    return X

def SVM_train(X, Y, learning_rate, num_iter=100, C=0, kernel='RBF', sigma=1):
    ''' Function that trains SVM on the labeled data (X,Y) '''
    n, m = X.shape
    c = Y.shape[0]
    
    # Define the hinge loss function
    def loss(X, Y, W, b, C):
        y = np.dot(W,X) + b
        reg = 0.5*np.sum(W**2)/X.shape[1]
        return np.sum(np.maximum(0, 1-Y*y))/X.shape[1] + reg/C
    
    # Transform the data using the kernel trick    
    if kernel == 'RBF':
        F = K_RBF(X, X, sigma)
    elif kernel == 'linear':
        F = K_linear(X)
        
    # Initialize parameters
    W = 0.01*np.random.rand(c,F.shape[0])
    b = np.zeros((c,1))

    for i in range(num_iter):
        y = np.matmul(W,F) + b
        dy = Y*np.maximum(0,np.sign(1-Y*y))
        db = -np.sum(dy, axis=1, keepdims=True)/m
        dW = -np.matmul(dy,F.T)/m + W/m/C
        W = W - learning_rate*dW
        b = b - learning_rate*db
        print('SVM loss is {}'.format(loss(F,Y,W,b,C)))
    return W, b

def SVM_predict(X_test, X_train, W, b, kernel='RBF', sigma=1):
    # Transform the data using the kernel trick    
    if kernel == 'RBF':
        F = K_RBF(X_train, X_test, sigma)
    elif kernel == 'linear':
        F = K_linear(X)
    
    # Compute class score and return class prediction
    y = np.dot(W,F) + b
#    return np.argmax(y, axis=0)
    return y

# Construct some data to test on
np.random.seed(2)
n_points = 100
X1 = np.random.randn(2,n_points//2) + 1.25*np.array([[-1],[1]])
Y1 = np.ones((1,n_points//2))
X2 = np.random.randn(2,n_points//2) + 1.25*np.array([[1],[-1]])
Y2 = -np.ones((1,n_points//2))
X = np.concatenate((X1,X2), axis=1)
Y = np.concatenate((Y1,Y2), axis=1)
permutation = np.random.permutation(X.shape[1])
X = X[:,permutation]
Y = Y[:,permutation]

# Plot the data
plt.figure('SVM test')
plt.clf()
plt.scatter(X1[0,:],X1[1,:])
plt.scatter(X2[0,:],X2[1,:])

# Train the SVM on the data
W, b = SVM_train(X, Y, .1, num_iter=10000, C=.001)

# Map the decision boundary and probability density
resolution = 30
x_range = np.linspace(-4,4,resolution)
y_range = np.linspace(-4,4,resolution)
Z = np.zeros((len(x_range), len(y_range)))
for idx, x in enumerate(x_range):
        X_test = np.array([x*np.ones(resolution), y_range])
        Z[idx,:] = SVM_predict(X_test, X, W, b)
plt.contour(x_range, y_range, Z, levels=[0])
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.draw()























