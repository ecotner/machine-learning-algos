# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:11:01 2018

@author: Eric Cotner
"""

# Import necessary modules
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import numpy as np
import utils as u

# Define hyperparameters, file paths, and control flow variables
BATCH_SIZE = 128
VAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SAVE_PATH = './checkpoints/CIFAR10_0'
MAX_EPOCHS = int(1e10)
CIFAR_DATA_PATH = './CIFAR10_data/'

# Load image data. Returns a dict that contains 'data' (10000x3072 numpy array of uint8's) and 'labels' (list of 10000 numbers between 0-9 denoting the class). There are 5 training data files, one test file, and one 'meta' file containing a list of human-decipherable names to the classes.
print('Loading data...')
data = []
for n in range(1,5+1):
    data.append(u.unpickle(CIFAR_DATA_PATH+'data_batch_'+str(n)))
data.append(u.unpickle(CIFAR_DATA_PATH+'test_batch'))
data.append(u.unpickle(CIFAR_DATA_PATH+'batches.meta'))

# Extract data from dict
print('Processing data...')
X = np.stack([data[n]['data'] for n in range(5)], axis=-1).reshape(shape=[-1,32,32,3])
Y = np.stack([data[n]['labels'] for n in range(5)], axis=-1)
X_test = data[5]['data'].reshape(shape=[-1,32,32,3])
Y_test = data[5]['labels']
label_names = data[6]
del data

# Preprocess data by normalizing to zero mean and unit variance
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean)/X_std
X_test = (X_test - X_mean)/X_std

# Shuffle data and split into training and validation sets
np.random.seed(0)
X = np.random.permutation(X)
np.random.seed(0)
Y = np.random.permutation(Y)
X_val = X[:VAL_BATCH_SIZE,:,:,:]
Y_val = Y[:VAL_BATCH_SIZE]
X_train = X[VAL_BATCH_SIZE:,:,:,:]
Y_train = Y[VAL_BATCH_SIZE:]
del X, Y
m_train = len(Y_train)
m_val = len(Y_val)
m_test = len(Y_test)

# Log some info about the data for future use
with open(SAVE_PATH+'.log', 'w+') as fo:
    fo.write('Training log\n\n')
    fo.write('Dataset metrics:\n')
    fo.write('Training data shape: {}\n'.format(X_train.shape))
    fo.write('Validation data shape: {}\n'.format(X_val.shape))
    fo.write('Test data shape: {}\n'.format(X_test.shape))
    fo.write('X_mean: {}\n'.format(X_mean))
    fo.write('X_std: {}\n\n'.format(X_std))
    fo.write('Hyperparameters:\nBatch size: {}\n'.format(BATCH_SIZE))
    fo.write('Learning rate: {}\n'.format(LEARNING_RATE))

# Load network architecture/parameters
G = u.load_graph(SAVE_PATH)
with G.as_default():
    saver = tf.train.Saver()
    
    # Get important tensors/operations from graph
    X = G.get_tensor_by_name('input:0')
    Y = G.get_tensor_by_name('output:0')
    labels = G.get_tensor_by_name('labels:0')
    J = G.get_tensor_by_name('loss:0')
    training_op = G.get_operation_by_name('training_op')
    learning_rate = G.get_tensor_by_name('learning_rate:0')
    
    # Load the preprocessed training and validation data in TensorFlow constants if possible so that there is no bottleneck sending things from CPU to GPU
    X_train = tf.constant(X_train)
    Y_train = tf.constant(Y_train)
#    X_val = tf.constant(X_val)
#    Y_val = tf.constant(Y_val)
#    del X_train, Y_train, X_val, Y_val
    
    # Reroute tensors to the location of the 
    train_idx = tf.placeholder(tf.int32, shape=[None])
    is_train = tf.placeholder(tf.int32, shape=[])
    X_train = tf.gather(X_train, train_idx)
    Y_train = tf.gather(Y_train, train_idx)
    ge.reroute_ts([X_train, Y_train], [X, labels])
    
    # Start the TF session and load variables
    with tf.Session() as sess:
        saver.restore(sess, SAVE_PATH)
        
        # Initialize control flow variables
        min_val_loss = np.inf
        
        # Iterate over epochs
        for epoch in range(MAX_EPOCHS):
            
            # Iterate over batches
            for b in range(m_train//BATCH_SIZE+1):
                
                # Perform forward/backward pass
                slice_lower = b*BATCH_SIZE
                slice_upper = min((b+1)*BATCH_SIZE, m_train)
                train_loss, _ = sess.run([J, training_op], feed_dict={learning_rate:LEARNING_RATE, train_idx:range(slice_lower, slice_upper)})
                
                # Compute metrics and add to logs
                with open(SAVE_PATH+'train_loss.log', 'a+') as fo:
                    fo.write(str(train_loss)+'\n')
                print('Epoch: {}, batch: {}/{}, loss: {}'.format(epoch, b, m_train//BATCH_SIZE, train_loss))
            
            # Save progress at end of epoch (if validation loss has improved)
            val_loss = sess.run(J, feed_dict={X_train:X_val})
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('Validation loss: {}'.format(min_val_loss))
                print('Saving variables...')
                saver.save(sess, SAVE_PATH)
            with open(SAVE_PATH+'val_loss.log', 'a+') as fo:
                fo.write(str(val_loss)+'\n')
        
        # Print stuff once done
        print('Done!')
























