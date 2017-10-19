# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:38:15 2017

A simple autoencoder that takes an arbitrary input and creates a lower-dimensional representation by passing it through a single hidden layer (with relu activations). Tested it on encoding a random binary string of length 10 into a 7-dim representation; seems to work pretty well, although it does have trouble with some of the digits. Some of the predictions give 0.4~0.5 (typically the 5th or 6th digit), which could go either way, but most are within ~0.01 of either 0 or 1. Perhaps a deeper network will be able to learn a more complex representation? Also, I'm pretty sure this problem is impossible from an information theory standpoint unless some additional assumptions are made (like the number of 1's is fixed), otherwise a bit string already contains the minimum amount of information.

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''========================== HELPER FUNCTIONS =========================='''

def make_minibatches(X, batch_size, batch_axis=0):
    batches = []
    m = X.shape[batch_axis]
    perm_idx = np.random.permutation(m)
    X_perm = X[perm_idx,:]
    for b in range(m//batch_size):
        minibatch = X_perm[b*batch_size:(b+1)*batch_size,:]
        batches.append(minibatch)
    if m % batch_size != 0:
        minibatch = X_perm[(b+1)*batch_size:,:]
        batches.append(minibatch)
    return batches

'''====================== SET UP COMPUTATIONAL GRAPH ===================='''

def initialize_graph(input_size, hidden_width):
    tf.reset_default_graph()
    G = tf.Graph()
    
    with G.as_default():
        
        # Input layer (takes arbitrary # of batches and features)
        X = tf.placeholder(tf.float32, [None,input_size], name='X')
        
        # Hidden layer
        W1 = tf.Variable(np.random.randn(input_size,hidden_width)*np.sqrt(2/(input_size+hidden_width)), dtype=tf.float32, name='W1')
        b1 = tf.Variable(np.zeros((hidden_width,)), dtype=tf.float32, name='b1')
        A1 = tf.nn.relu(tf.matmul(X,W1)+b1, name='A1')
        
        # Output layer
        W2 = tf.Variable(np.random.randn(hidden_width,input_size)*np.sqrt(2/(hidden_width+input_size)), dtype=tf.float32, name='W2')
        b2 = tf.Variable(np.zeros((input_size,)), dtype=tf.float32, name='b2')
        Y = tf.sigmoid(tf.matmul(A1,W2)+b2, name='Y')
        
        # Loss function: MSE of difference between input and output
        loss = tf.reduce_mean(-X*tf.log(Y+1e-20) - (1-X)*tf.log(1-Y+1e-20), name='loss')
    # Return the computational graph
    return G

'''=========================== TRAIN THE NETWORK =========================='''

def train(X_train, hidden_width, lr, max_epochs, batch_size=32, X_val=None, restore_from_checkpoint=False, plot_every_n_steps=100, save_every_n_epochs=1):
    ''' Trains the autoencoder. Assumes X has shape (n_examples,n_features). '''
    n_examples, n_features = X_train.shape
    save_path = './checkpoints/SimpleAutoencoder_1'
    
    # Make minibatches from data
    X_train_batches = make_minibatches(X_train, batch_size)
#    if X_val is not None:
#        X_val_batches = make_minibatches(X_val, batch_size)
    
    # Set up the computational graph
    G = initialize_graph(n_features, hidden_width)
    with G.as_default():
        # Extract useful tensors from Graph
        X_input = G.get_tensor_by_name('X:0')
        loss = G.get_tensor_by_name('loss:0')
        Y = G.get_tensor_by_name('Y:0')
        #  Define training op, Saver, and TF session
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)
        saver = tf.train.Saver(var_list=None)
        with tf.Session() as sess:
            # Reload variables or start fresh
            if restore_from_checkpoint == True:
                saver.restore(sess, save_path)
            else:
                sess.run(tf.global_variables_initializer())
            # Iterate over epochs
            global_step = 0
            batch_loss_list = []
            step_list = []
            val_loss_list = []
            for ep in range(max_epochs):
                # Iterate over minibatches
                for b, batch in enumerate(X_train_batches):
                    global_step += 1
                    # Calculate loss and do training op
                    _, batch_loss = sess.run([train_op, loss], feed_dict={X_input:batch})
                    print('Epoch {}/{}, batch {}/{}, loss: {}'.format(ep+1, max_epochs, b+1, len(X_train_batches), batch_loss))
                    # Plot batch loss
                    if global_step % plot_every_n_steps == 1:
                        batch_loss_list.append(batch_loss)
                        step_list.append(global_step/len(X_train_batches))
                        plt.figure('Loss')
                        plt.clf()
                        plt.semilogy(step_list, batch_loss_list, label='Training')
                        if X_val is not None:
                            val_loss = sess.run(loss, feed_dict={X_input:X_val})
                            val_loss_list.append(val_loss)
                            plt.semilogy(step_list, val_loss_list, label='Validation')
                        plt.title('Batch loss')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        plt.legend()
                        plt.draw()
                        plt.pause(1e-10)
                # Save progress
                if ep % save_every_n_epochs == 0:
                    print('Saving...')
                    saver.save(sess, save_path)
            print('Saving...')
            saver.save(sess, save_path)
            print('Training complete!')

def model_run(n_examples, n_features, n_ups, hidden_width):
    X = generate_data(n_examples, n_features, n_ups)
    save_path = './checkpoints/SimpleAutoencoder_1'
    
    # Set up the computational graph
    G = initialize_graph(n_features, hidden_width)
    with G.as_default():
        # Extract useful tensors from Graph
        X_input = G.get_tensor_by_name('X:0')
        loss = G.get_tensor_by_name('loss:0')
        Y_output = G.get_tensor_by_name('Y:0')
        #  Define Saver
        saver = tf.train.Saver(var_list=None)
        with tf.Session() as sess:
            # Recover from last checkpoint
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, save_path)
            Y = sess.run(Y_output, feed_dict={X_input:X})
            for i in range(X.shape[0]):
                print('Original data: \t\t{}'.format(X[i,:].astype(int)))
                print('Autoencoded data: \t{}'.format((Y[i,:]>0.5).astype(int)))
                q = input('Press enter for another, or q to quit: ')
                if q.lower() == 'q':
                    break
            
''' ======================== PREPROCESS DATA ============================== '''

def generate_data(n_examples, n_features=10, n_ups=1):
    ''' Generates fake data which is simply <n_ups> randomly placed up bits (1) in an array of zeros of total length <array_len>. '''
    array = np.zeros((n_examples, n_features))
    for m in range(n_examples):
        perm_idx = np.random.permutation(n_features)[:n_ups]
        array[m,perm_idx] = 1
    return array

''' ======================= TEST IT OUT ============================= '''

n_features = 10
n_ups = 5
hidden_width = 7
#X = generate_data(10**4, n_features, n_ups)
#X_val = generate_data(100, n_features, n_ups)
#train(X, hidden_width, lr=1e-3, max_epochs=10**3, batch_size=32, X_val=X_val, restore_from_checkpoint=False, plot_every_n_steps=100, save_every_n_epochs=100)

model_run(100, n_features, n_ups, hidden_width)



























