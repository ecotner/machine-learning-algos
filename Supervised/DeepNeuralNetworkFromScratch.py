# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:15:33 2017

Module for creating fully-connected deep neural network from scratch.
Implementation of backprop is done with analytically-computed derivatives, and
is automatically taken care of within the training method, which uses a gradient
descent optimizer with momentum.

The network architecture can be arbitrarily deep/wide (in theory), and supports
a number of hidden unit activation functions, such as 'tanh', 'ReLU', 'LReLU'
(leaky ReLU) and 'sigmoid'.

The activation of the output layer can be either 'tanh', 'sigmoid', 'linear' or
'softmax', and this feeds directly into the cost function, which is of either a
'cross-entropy' (for classification) or 'MSE' (for regression) type. It is
recommended to use the 'softmax' or 'sigmoid' output in conjunction with
the 'cross-entropy' cost function, and 'linear' output in conjunction with 'MSE'.

It also supports 'L2', 'L1', and (inverted) 'dropout' regularization, where the
regularization parameter lambda is allowed to vary from layer to layer.

TO DO: Implement RMSprop and ADAM (i.e. update momentum to include RMS)

TO DO: Implement learning rate decay (give option for a function alpha, which
returns the learning rate as a function of epoch number).

@author: 27182_000
"""

import numpy as np
import matplotlib.pyplot as plt
EPSILON = 1e-100

def softmax(Z):
        Z_max = np.max(Z, axis=0, keepdims=True)
        exp = np.exp(Z-Z_max)
        return exp/np.sum(exp, axis=0)

''' ================== NEURAL NETWORK CLASS ======================== '''
class NeuralNetwork(object):
    '''
    Defines a neural network class which will encapsulate all the different
    layers and computations, including forward prop and backprop.
    Args:
        architecture: tuple of integers which specify the width of the network
        in each layer. I.e. (a,b,c) indicates a network with an input layer of
        width a, hidden layer of width b, and output layer of width c.
        activation: The type of activation function to use in the hidden layers.
        Choices include 'tanh', 'ReLU' and 'sigmoid'.
        output: the activation of the output layer. Choices are 'tanh', 'sigmoid',
        'linear', and 'softmax'.
        cost_function: the choice of cost function to use. Possible choices are
        'cross-entropy' and 'MSE' (Mean Squared Error).
    '''
    
    def __init__(self, architecture=(1,1), activation='tanh', output='softmax',
                 cost_function='cross-entropy'):
        if output == 'softmax' and architecture[-1] == 1:
            output = 'sigmoid'
        self.architecture = architecture
        self.layers = ([InputLayer(architecture[0])] + 
                       [HiddenLayer(w, activation) for w in architecture[1:-1]] +
                       [OutputLayer(architecture[-1], output, cost_function)])     # List of layers in the network
        self.weights = [None] + [np.sqrt(2/self.layers[l-1].width)*np.random.randn(self.layers[l].width, self.layers[l-1].width) for l in range(1,len(self.layers))]    # Uses He initialization
        self.biases = [None] + [np.zeros((self.layers[l].width, 1), dtype=float) for l in range(1,len(self.layers))]
        self.dweights = [None] + [np.zeros((architecture[l], architecture[l-1])) for l in range(1,len(self.layers))]
        self.dbiases = [None] + [np.zeros((architecture[l], 1)) for l in range(1,len(self.layers))]
        self.prev_weights = None
        self.prev_biases = None
        self.normalization = (None, None)
        
    def initialize_parameters(self):
        self.prev_weights = self.weights.copy()
        self.prev_biases = self.biases.copy()
        self.weights = [None] + [np.sqrt(2/self.layers[l-1].width)*np.random.randn(self.layers[l].width, self.layers[l-1].width) for l in range(1,len(self.layers))]    # Uses He initialization
        self.biases = [None] + [np.zeros((self.layers[l].width, 1), dtype=float) for l in range(1,len(self.layers))]
        self.dweights = [None] + [np.zeros((self.architecture[l], self.architecture[l-1])) for l in range(1,len(self.layers))]
        self.dbiases = [None] + [np.zeros((self.architecture[l], 1)) for l in range(1,len(self.layers))]
    
    def forward_propagation(self, X, regularization=None, lambda_=0):
        '''
        Does a single pass of forward propagation.
        Args:
            X: Input design matrix. X.shape = (num_features, num_examples)
            regularization: type of regularization. Only 'dropout' has an effect
            on forward propagation
            lambda: if using 'dropout' regularization, this is the probability
            to keep a node. If it is a scalar, this probability is applied to
            every layer, but if it is a list, then each element is the keep
            probability for each individual layer.
        Output:
            A_list: List of the values of the activation functions.
            Z_list: List of the values of the input to the activation functions.
        '''
        A_list = [X.copy()]
        Z_list = [None]
        A = X.copy()
        for l in range(1,len(self.layers)):
            Z = np.dot(self.weights[l],A) + self.biases[l]
            A = self.layers[l].a(Z)
            if regularization == 'dropout' and l != len(self.layers)-1:
                if type(lambda_) == int or type(lambda_) == float:
                    lambda_temp = lambda_*np.ones(len(self.layers))
                A *= (np.random.rand(A.shape[0], A.shape[1]) < lambda_temp[l])/lambda_temp[l]
            A_list.append(A)
            Z_list.append(Z)
        return Z_list, A_list
    
    def backward_propagation(self, Z_list, A_list, Y, m, regularization, lambda_):
        '''
        Does a single pass of backward propagation and returns the gradient with
        respect to the weights and biases.
        Args:
            Z_list, A_list: Lists of the activation function input and output
            matrices.
            Y: Matrix of training example labels
            regularization: The type of regularization to use. Possible choices
            include None, 'L2' and 'L1'.
            lambda_: Regularization parameter. If regularization='dropout', this
            is the probability to keep a node.
        Output:
            dW_list: List of the gradient of the cost function with respect to
            the weights in each layer.
            db_list: List of the gradient of the cost function with respect to
            the biases in each layer.
        '''
        m_Y = Y.shape[1]
        if type(lambda_) == int or type(lambda_) == float:
            lambda_temp = lambda_*np.ones(len(self.layers))
        if self.layers[-1].cost_function == 'cross-entropy':
            g = ((1-Y)/(1-A_list[-1]+EPSILON) - Y/(A_list[-1]+EPSILON))
        elif self.layers[-1].cost_function == 'MSE':
            g = A_list[-1] - Y
        else:
            raise Exception('unknown cost function in backprop algorithm')
        db_list, dW_list = ([], [])
        for l in range(len(self.layers)-1,0,-1):
            g = g * self.layers[l].aprime(Z_list[l])
            if regularization == 'L2':
                db_reg = lambda_temp[l]*self.biases[l]/m
                dW_reg = lambda_temp[l]*self.weights[l]/m
            elif regularization == 'L1':
                db_reg = lambda_temp[l]*np.sign(self.biases[l])/m
                dW_reg = lambda_temp[l]*np.sign(self.weights[l])/m
            elif regularization == 'dropout':
                g *= np.sign(A_list[l])
                db_reg = 0
                dW_reg = 0
            else:
                db_reg = 0
                dW_reg = 0
            db = np.sum(g, axis=1, keepdims=True)/m_Y + db_reg
            dW = np.dot(g, A_list[l-1].T)/m_Y + dW_reg
            g = np.dot(self.weights[l].T, g)
            db_list.append(db)
            dW_list.append(dW)
        db_list.append(None)
        dW_list.append(None)
        return dW_list[::-1], db_list[::-1]
    
    def update_parameters(self, dW_list, db_list, learning_rate, momentum):
        '''
        Updates the weights and biases of the network after the completion of
        backprop. This implementation supports 'momentum', which uses an
        exponentially-weighted average over the previous gradients to compute the
        update.
        Args:
            dW_list: List of the gradients of the weights computed from backprop.
            db_list: List of gradients of biases.
            learning_rate: The proportionality factor of the cost function
            gradient by which the parameters are updated.
            momentum: Parameter between 0 and 1 which tells how many previous
            gradients to average over, 0 being none and 1 being all of them.
        '''
        for l in range(1,len(self.layers)):
            if momentum == 0 or np.all(self.dweights[l] == 0):
                self.dweights[l] = dW_list[l]
                self.dbiases[l] = db_list[l]
            if 0 < momentum <=1:
                self.dweights[l] = momentum*self.dweights[l] + (1-momentum)*dW_list[l]
                self.dbiases[l] = momentum*self.dbiases[l] + (1-momentum)*db_list[l]
            elif momentum == 0:
                pass
            else:
                raise Exception('momentum need to be between 0 and 1')
            self.weights[l] -= learning_rate * self.dweights[l]
            self.biases[l] -= learning_rate * self.dbiases[l]
    
    def cost(self, Y, Yhat, m, regularization, lambda_):
        assert Y.shape == Yhat.shape
        n_classes, m_Y = Y.shape
        if self.layers[-1].cost_function == 'cross-entropy':
            cost_ = -np.sum(Y*np.log(Yhat+EPSILON) + (1-Y)*np.log(1-Yhat+EPSILON))/m_Y
        elif self.layers[-1].cost_function == 'MSE':
            cost_ = 0.5*np.sum((Y-Yhat)**2)/m_Y
        else:
            assert Exception('unknown cost function')
        cost_reg = 0
        if regularization == 'L2':
            for l in range(1,len(self.layers)):
#                pass
                cost_reg += (0.5*lambda_/m)*np.sum(self.weights[l]**2)
        elif regularization == 'L1':
            for l in range(1,len(self.layers)):
                cost_reg += (lambda_/m)*np.sum(np.abs(self.weights[l]))
        return cost_ + cost_reg
    
    def train(self, X, Y, num_epoch, batch_size=None, regularization=None, lambda_=0,
              learning_rate=1.0, momentum=0, print_progress=True,
              plot_progress=False, X_val=None, Y_val=None, replay=False):
        if type(num_epoch) != int:
            num_epoch = int(num_epoch)
        if replay == False:
            self.initialize_parameters()
        # Normalize data
        X_mu = np.mean(X, axis=1, keepdims=True)
        X_var = np.var(X, axis=1, keepdims=True)
        self.normalization = (X_mu, X_var)
        X_norm = (X-X_mu)/X_var
        if X_val is not None:
            X_val_norm = (X_val-X_mu)/X_var
        # Split training set up into mini-batches
        n_X, m = X_norm.shape
        if batch_size == None:
            X_batches = [X_norm]
            Y_batches = [Y]
        elif type(batch_size) == int:
            permutation = list(np.random.permutation(m))
            X_shuffled = X_norm[:,permutation]
            Y_shuffled = Y[:,permutation]
            X_batches = [X_shuffled[:,i:i+batch_size] for i in range(0, m-batch_size, batch_size)]
            Y_batches = [Y_shuffled[:,i:i+batch_size] for i in range(0, m-batch_size, batch_size)]
            if m % batch_size != 0:
                X_batches.append(X_shuffled[:,batch_size*(m//batch_size):])
                Y_batches.append(Y_shuffled[:,batch_size*(m//batch_size):])
        else:
            raise Exception('batch_size must be an integer')
        # Set up some stuff on how often to give progress updates
        if print_progress == True:
            print_progress = len(X_batches)*num_epoch//100
            if print_progress == 0:
                print_progress = 1
        elif print_progress == False:
            print_progress = len(X_batches)*num_epoch + 100 # Ensures that if print_progress is False, that it never triggers, but avoids the issue of modding a number by zero
        if plot_progress == True:
            plot_progress = print_progress
        elif plot_progress == False:
            plot_progress = len(X_batches)*num_epoch + 100  # Ensures that if plot_progress is False, that it never triggers, but avoids the issue of modding a number by zero
        # Set up training cost plot
        iter_list = []
        cost_list = []
        val_cost_list = []
        if plot_progress != False:
            plt.ion()
            plt.figure('Training cost')
        # Run forwardprop -> backprop -> update weights loop repeatedly
        total_iter = num_epoch*len(X_batches)
        for e in range(num_epoch):
            for b in range(len(X_batches)):
                current_iter = e*len(X_batches) + b
                # The actual calculation:
                Z_list, A_list = self.forward_propagation(X_batches[b], regularization, lambda_)
                dW_list, db_list = self.backward_propagation(Z_list, A_list, Y_batches[b], m, regularization, lambda_)
                cost = self.cost(Y_batches[b], A_list[-1], m, regularization, lambda_)
                # Plots progress of cost function
                if print_progress == plot_progress == False:
                    pass
                elif current_iter % print_progress == 0 or current_iter % plot_progress == 0:
                    # Plots just training cost if no validation set provided
                    if X_val is None and Y_val is None:
                        if current_iter % print_progress == 0:
                            print('Progress: {}/{} ({:.1f}%), epoch={}. Train cost={:.3e}\n'.format(current_iter, total_iter, 100*current_iter/total_iter, e+1, cost))
                        if current_iter % plot_progress == 0:
                            iter_list.append(current_iter)
                            cost_list.append(cost)
                            plt.cla()
                            plt.semilogy(iter_list, cost_list, label='Training')
                    # Plots both training/validation cost if both sets provided
                    elif X_val is not None and Y_val is not None:
                        val_Z_list, val_A_list = self.forward_propagation(X_val_norm, regularization, lambda_)
                        val_cost = self.cost(Y_val, val_A_list[-1], m, regularization, lambda_)
                        if current_iter % print_progress == 0:
                            print('Progress: {}/{} ({:.1f}%), epoch={}. Train cost={:.3e}, val cost={:.3e}\n'.format(current_iter, total_iter, 100*current_iter/total_iter, e+1, cost, val_cost))
                        if current_iter % plot_progress == 0:
                            iter_list.append(current_iter)
                            cost_list.append(cost)
                            val_cost_list.append(val_cost)
                            plt.cla()
                            plt.semilogy(iter_list, cost_list, label='Training')
                            plt.semilogy(iter_list, val_cost_list, label='Validation')
                    if current_iter % plot_progress == 0:
                        plt.xlabel('Num. iterations')
                        plt.ylabel('Cost')
                        plt.title('Cost while training')
                        plt.xlim((0, total_iter))
#                        plt.ylim((0,max(cost_list + val_cost_list)))
                        plt.legend()
                        plt.draw()
                        plt.pause(1e-9)
                # Updates weights/biases provided cost isn't bad
                if np.isnan(cost):
                    print('Cost function under/overflow; exiting training loop.')
                    break
                else:
                    self.update_parameters(dW_list, db_list, learning_rate, momentum)
        # Prints final cost of whole training set
        if not np.isnan(cost):
            Z_list, A_list = self.forward_propagation(X_norm, regularization, lambda_)
            cost = self.cost(Y, A_list[-1], m, regularization, lambda_)
            print('Progress: {}/{} ({:.1f}%). Final training cost={:.3e}\n'.format(total_iter, total_iter, 100, cost))
        if plot_progress != False:
            plt.ioff()
    
    def predict(self, X):
        Z_list, A_list = self.forward_propagation((X-self.normalization[0])/self.normalization[1])
        Y_pred = A_list[-1]
        return Y_pred
    
    def evaluate_classifier(self, X_train, Y_train, X_val, Y_val):
        if self.layers[-1].cost_function != 'cross-entropy':
            raise Exception('Can\'t evaluate non-classifier model.')
        Y_train_pred = self.predict(X_train)>0.5
        Y_val_pred = self.predict(X_val)>0.5
        train_accuracy = np.mean((Y_train_pred>0.5) == Y_train)
        val_accuracy = np.mean((Y_val_pred>0.5) == Y_val)
        print('Training error: {}'.format(1-train_accuracy))
        print('Validation error: {}'.format(1-val_accuracy))
        n_out, m_val = Y_val.shape
        if n_out == 1:  # Might need to fix this for networks with single output node
            # Calculate number of true/false positives/negatives
            tp = np.sum((Y_val_pred == 1) and (Y_val == 1))
            fp = np.sum((Y_val_pred == 1) and (Y_val == 0))
            tn = np.sum((Y_val_pred == 0) and (Y_val == 0))
            fn = np.sum((Y_val_pred == 0) and (Y_val == 1))
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            F1 = 2*precision*recall/(precision+recall)
            print('Following statistics based off validation set with m_val={}'.format(m_val))
            print('True positives: {}'.format(tp))
            print('False positives: {}'.format(fp))
            print('True negatives: {}'.format(tn))
            print('False negatives: {}'.format(fn))
            print('F1 score: {}'.format(F1))
    
    def __str__(self):
        msg = 'A fully-connected neural network of depth L={} with the following architecture:\n'.format(len(self.layers)-1)
        for l in range(len(self.layers)):
            msg += 'Layer {}: {} activation with {} nodes\n'.format(l, self.layers[l].activation, self.layers[l].width)
        return msg

''' =================== LAYER CLASSES ============================ '''
class Layer(object):
    pass
    
class InputLayer(Layer):
    def __init__(self, width=1):
        self.width = width
        self.activation = 'input'

class HiddenLayer(Layer):
    def __init__(self, width=1, activation='tanh'):
        self.width = width
        self.activation = activation
    
    def a(self, Z):
        if self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif self.activation == 'ReLU':
            return np.maximum(0, Z)
        elif self.activation == 'LReLU':
            return np.maximum(0.01*Z, Z)
        else:
            raise Exception('unknown activation function')
    
    def aprime(self, Z):
        if self.activation == 'tanh':
            return np.cosh(Z)**(-2)
        elif self.activation == 'sigmoid':
            return 0.25/np.cosh(Z/2)**2
        elif self.activation == 'ReLU':
            return 1*(Z>0)
        elif self.activation == 'LReLU':
            return 1*(Z>0) + 0.01*(Z<=0)
        else:
            raise Exception('unknown activation function')

class OutputLayer(Layer):
    def __init__(self, width=1, output='sigmoid', cost_function='cross-entropy'):
        self.width = width
        self.activation = output
        self.cost_function = cost_function
    
    def a(self, Z):
        if self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-Z))
        elif self.activation == 'linear':
            return Z
        elif self.activation == 'softmax':
            return softmax(Z)
        else:
            raise Exception('unknown output function')
    
    def aprime(self, Z):
        if self.activation == 'tanh':
            return np.cosh(Z)**(-2)
        elif self.activation == 'sigmoid':
            return 0.25/np.cosh(Z/2)**2
        elif self.activation == 'linear':
            return 1
        elif self.activation == 'softmax':
            return softmax(Z)*(1-softmax(Z))
        else:
            raise Exception('unknown output function')

def XOR_UnitTest():
    ''' Tests the neural network out by having it learn the XOR function '''
    n_in, n_hidden, n_out, m = (2, 10, 1, 100)
    train_frac = 0.8        # Percentage of examples used in training set
    NN = NeuralNetwork((n_in,n_hidden,n_out), activation='ReLU', output='sigmoid', cost_function='cross-entropy')
    X = np.random.randn(n_in,m)>0
    Y = 1.0*np.logical_xor(X[0,:], X[1,:]).reshape((n_out, m))
    X_train, Y_train = (X[:,:int(m*train_frac)], Y[:,:int(m*train_frac)])
    X_val, Y_val = (X[:,int(m*train_frac):], Y[:,int(m*train_frac):])
    NN.train(X_train, Y_train, 10000, learning_rate=0.3, print_progress=1000)
    Y_train_pred = NN.predict(X_train)
    Y_val_pred = NN.predict(X_val)
    train_accuracy = np.mean((Y_train_pred>0.5) == Y_train)
    val_accuracy = np.mean((Y_val_pred>0.5) == Y_val)
    print('Model predicts correct output {} percent of time on training set'.format(100*train_accuracy))
    print('Model predicts correct output {} percent of time on validation set'.format(100*val_accuracy))
#    print(Y_train_pred)
#    print(Y_train)
    return NN

def Quadrant_UnitTest():
    ''' Test the neural network by having it classify points in a 2D plane based
    on which quadrant they are in. '''
    # Set up network architecture and hyperparamters
    NN_architecture = (2, 20, 10, 5, 4)
    n_in = NN_architecture[0]
    n_out = NN_architecture[-1]
    m = 1000
    n_train = int(0.9*m)
    # Generate data, split into training/validation sets
    X = np.random.randn(n_in,m)
    Y = np.zeros((n_out, m), dtype=int)
    for i in range(m):
        if X[0,i] > 0 and X[1,i] > 0:
            Y[0,i] = 1
        elif X[0,i] < 0 and X[1,i] > 0:
            Y[1,i] = 1
        elif X[0,i] < 0 and X[1,i] < 0:
            Y[2,i] = 1
        elif X[0,i] > 0 and X[1,i] < 0:
            Y[3,i] = 1
    X_train, Y_train = (X[:,:n_train], Y[:,:n_train])
    X_val, Y_val = (X[:,n_train:], Y[:,n_train:])
    # Plot the data points according to their quadrant
    plt.figure('quadrants')
    plt.clf()
    for q in range(n_out):
        plt.scatter(X[0,Y[q,:]==1],X[1,Y[q,:]==1])
    plt.show()
    # Set up and train the neural network
    NN = NeuralNetwork(NN_architecture, activation='LReLU',
                       output='softmax', cost_function='cross-entropy')
    print(NN)
    NN.train(X_train, Y_train, num_epoch=1e4, regularization='L1', lambda_=0.03,
             learning_rate=0.9, momentum=.9, batch_size=250, print_progress=True,
             plot_progress=True, X_val=X_val, Y_val=Y_val, replay=False)
    NN.evaluate_classifier(X_train, Y_train, X_val, Y_val)
    
    
Quadrant_UnitTest()
#NN = XOR_UnitTest()







