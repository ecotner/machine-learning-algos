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

file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

tf.logging.set_verbosity(tf.logging.INFO)

# Import the dataset
train_raw = pd.read_csv('./train.csv')

# Get the data into numpy-readable form
Y = np.array(train_raw.iloc[:,0]) # Will convert to one-hot tensor later
X = np.array(train_raw.iloc[:,1:])/255
m, n_0 = X.shape  # Should be 42000 examples of 784 features (28x28 pixels)
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

# Build the neural network
def cnn_model(features, labels, mode):
    ''' Model for building the CNN '''
    
    # Define architecture
    input_layer = tf.reshape(tf.cast(features['x'], tf.float32), [-1,28,28,1])
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=(5,5),
            padding='same',
            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=64,
            padding='same',
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
                                training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'classes': tf.argmax(input=logits, axis=1),
            # Add 'softmax_tensor' to the graph. It is used for PREDICT and by
            # the 'logging_hook'.
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.Variable(labels, trainable=False, dtype=tf.float64)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    # Configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=np.argmax(labels, axis=1),
                                            predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




'''======================= TESTING, ATTENTION PLEASE ======================'''
#plot_samples(X,Y,idx=range(X.shape[0]))

# Create the classifier from the model
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir='/tmp/mnist_convnet_classifier')

# Create logging hook
#tensors_to_log = {'probabilities': 'softmax_tensor'}
#logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_secs=2)

# Put data in input functions
input_fn = tf.estimator.inputs.numpy_input_fn({'x':X_train}, Y_train, batch_size=128, num_epochs=None,
                                              shuffle=False)
train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':X_train}, Y_train, batch_size=128, num_epochs=10,
                                              shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':X_val}, Y_val, batch_size=128, num_epochs=10,
                                                  shuffle=False)
# Train the network
mnist_classifier.train(input_fn=input_fn, steps=1000#, hooks=[logging_hook]
                       )

# Evaluate how well the model did
train_metrics = mnist_classifier.evaluate(input_fn=train_input_fn)
eval_metrics = mnist_classifier.evaluate(input_fn=eval_input_fn)
print('Train metrics: %r'% train_metrics)
print('Eval metrics: %r'% eval_metrics)




































