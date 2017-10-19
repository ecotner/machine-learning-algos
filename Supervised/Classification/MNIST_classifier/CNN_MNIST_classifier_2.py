# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:41:12 2017

Convolutional neural network for classifying MNIST handwritten digits.

@author: Eric Cotner
"""

# Import necessary modules
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Make sure current working directory is same as file directory
file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

# Enable logging
tf.logging.set_verbosity(tf.logging.INFO)

# Import the dataset
train_raw = pd.read_csv('./train.csv')

# Get the data into numpy-readable form
Y = np.array(train_raw.iloc[:,0]) # Will convert to one-hot tensor later
X = np.array(train_raw.iloc[:,1:])/255
m, n_0 = X.shape  # Should be 42000 examples of 784 features (28x28 pixels)
np.random.seed(seed=0) # Set the seed for reproducibility
permutation = np.random.permutation(m)
Y = Y[permutation]
X = X[permutation,:]
# Convert labels into one-hot vectors
Y_one_hot = np.zeros((m,10))
for idx in range(m):
    label = Y[idx]
    Y_one_hot[idx,label] = 1
# Split dataset into train/val
Y_train = Y_one_hot[:40000,:]
X_train = X[:40000,:]
Y_val = Y_one_hot[40000:,:]
X_val = X[40000:,:]

# Show some samples from the data
def plot_samples(X, Y=None, idx=None):
    ''' Plots the digits, with labels Y if given. Y is assumed to be a one-hot
    vector, not an actual label. '''
    if idx is None:
        idx_list = (np.random.randint(),)
    elif type(idx) == int:
        idx_list = (idx,)
    else:
        idx_list = idx
    plt.figure('MNIST')
    plt.ion()
    for i in idx_list:
        plt.clf()
        plt.imshow(X[i,:].reshape(28,28), cmap='gray_r')
        if Y is not None:
            plt.title('y_{}={}'.format(i, Y[i]))
        plt.draw()
        plt.pause(1e-9)
        q = input('Press any key for next image, or q to quit >')
        if q.lower() == 'q':
            break

#plot_samples(X, Y, idx=np.random.permutation(X.shape[0]))

# Build the neural network model
def cnn_model(features, labels, mode, params):
    ''' The model of the CNN.
    Args:
        features: A dict containing the design matrix and any other inputs
        features, with dimensions (num_examples, num_features)
        labels: The labels matrix, with dimensions (num_examples, num_classes)
        mode: determines what mode the classifier is operating in (TRAIN, PREDICT,
        or EVAL)
        params: input hyperparameters 'learning_rate' and 'keep_prob'
    Outputs:
        outputs a tf.estimator.EstimatorSpec instance
    '''
    
    # Create the network
    
    # Define some functions for initializing the weights/biases/layers
    def weight_variable(shape, name):
        W_initial = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        return tf.Variable(W_initial)
    def bias_variable(shape, name):
        b_initial = tf.constant(0.1, shape=shape)
        return tf.Variable(b_initial)
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    # Input layer, reshaped to 2D image
    input_layer = tf.reshape(tf.cast(features['X'], dtype=tf.float32), [-1,28,28,1])
        
    # Define first convolution layer
    W_conv1 = weight_variable([5,5,1,32], 'W_conv1') # 32 5x5x1 filters
    b_conv1 = bias_variable([32], 'b_conv1')
    conv1 = tf.nn.relu(conv2d(input_layer, W_conv1) + b_conv1)
    pool1 = max_pool_2x2(conv1) # pool1 layer has shape 14x14x32
    
    # Define second convolution layer
    W_conv2 = weight_variable([5,5,32,64], 'W_conv2') # 64 5x5x32 filters
    b_conv2 = bias_variable([64], 'b_conv2')
    conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
    pool2 = max_pool_2x2(conv2) # Pool2 layer has shape 7x7x64
    
    # Flatten pool layer and feed into dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    W_dense3 = weight_variable([7*7*64, 1024], 'W_dense3')
    b_dense3 = bias_variable([1024], 'b_dense3')
    dense3 = tf.nn.relu(tf.matmul(pool2_flat, W_dense3) + b_dense3)
    
    # Apply dropout before output layer
    keep_prob = tf.constant(params['keep_prob'], tf.float32)
    dropout4 = tf.nn.dropout(dense3, keep_prob)
    
    # Final output layer
    W_out5 = weight_variable([1024, 10], 'W_out5')
    b_out5 = bias_variable([10], 'b_out5')
    y_out = tf.matmul(dropout4, W_out5) + b_out5 # This is NOT the class prediction; have to take softmax later
    
    # Decide what to do if used in PREDICT, TRAIN or EVAL modes
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'label_pred': tf.argmax(tf.nn.softmax(y_out, dim=1), axis=1)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        # Define loss function and optimization routine
        xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, 
                                                                          labels=labels, 
                                                                          dim=1))
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(xentropy, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=xentropy, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, 
                                                                          labels=labels, 
                                                                          dim=1))
        predictions = {'label_pred': tf.argmax(tf.nn.softmax(y_out, dim=1), axis=1)}
        accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), 
                                             predictions['label_pred'])
#        val_accuracy = tf.metrics.accuracy(tf.argmax(labels['Y_val'], axis=1), 
#                                             predictions['label_pred'])
        eval_metric_ops = {'accuracy':accuracy}
        return tf.estimator.EstimatorSpec(mode=mode, loss=xentropy,
                                          eval_metric_ops=eval_metric_ops)

''' ====================== IMPLEMENTING MODEL ====================== '''

# Implement the model
model_params = {'learning_rate':1e-3, 'keep_prob':0.4}
MNIST_CNN = tf.estimator.Estimator(model_fn=cnn_model, 
                                   params=model_params, 
                                   model_dir='/tmp/MNIST_CNN')

# Set up inputs for both train/val sets
train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X':X_train},
        y=Y_train,
        num_epochs=None,
        shuffle=False)
train_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X':X_train},
        y=Y_train,
        num_epochs=1,
        shuffle=False)
val_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X':X_val},
        y=Y_val,
        num_epochs=1,
        shuffle=False)

# Train network
#MNIST_CNN.train(input_fn=train_input_fn, steps=1000)

ev = MNIST_CNN.evaluate(input_fn=val_input_fn)
print('Loss: {}'.format(ev['loss']))
print('Validation accuracy: {}'.format(ev['accuracy']))























