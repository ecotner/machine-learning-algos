# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 22:42:55 2017

Simple neural network for classifying handwritten digits from the MNIST dataset.
Dataset is provided from Kaggle.
Architecture:
    3-layer network with relu hidden activations, sigmoid output
Cost function:
    Softmax over 10 classes, plus an L2 regularizer on the weights
Optimization:
    Adam with beta2=0.99, exponential learning rate decay, batch_size=1000

TO DO:
    Add convolutional layers
    Plot train/val accuracy as function of time

@author: Eric Cotner
"""

# Import necessary modules
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random

file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

# Import the dataset
#train_raw = pd.read_csv('./train.csv')

# Get the data into numpy-readable form
Y = np.array(train_raw.iloc[:,0]) # Will convert to one-hot tensor later
X = np.array(train_raw.iloc[:,1:]).T/255
n_0, m = X.shape  # Should be 42000 examples of 784 features (28x28 pixels)
permutation = np.random.permutation(m)
Y = Y[permutation]
X = X[:,permutation]
# Convert labels into one-hot vectors
Y_one_hot = np.zeros((10,m))
for idx in range(m):
    label = Y[idx]
    Y_one_hot[label,idx] = 1
# Split dataset into train/val
Y_train = Y_one_hot[:,:40000]
X_train = X[:,:40000]
Y_val = Y_one_hot[:,40000:]
X_val = X[:,40000:]

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
        plt.imshow(X[:,i].reshape(28,-1), cmap='gray_r')
        if Y is not None:
            plt.title('y_{}={}'.format(i, Y[i]))
        plt.draw()
        plt.pause(1e-9)
        q = input('Press any key for next image, or q to quit >')
        if q.lower() == 'q':
            break

# Build the neural network
def NeuralNetwork(X, Y, init_learning_rate, num_epochs, batch_size=None, lambd=0, X_val=None, Y_val=None, X_test=None):
    # Define a leaky relu function
#    def lrelu(T, alpha=0.01, name='lrelu'):
#        tf.maximum(alpha*T, T, name=name)
    
    # Reset graph
    tf.reset_default_graph()
    
    # Create mini-batches
    n_0, m = X.shape
    minibatches = []
    if batch_size is not None:
        for k in range(m//batch_size):
            X_batch = X[:,k*batch_size:(k+1)*batch_size-1]
            Y_batch = Y[:,k*batch_size:(k+1)*batch_size-1]
            minibatches.append((X_batch, Y_batch))
        if m % batch_size != 0:
            X_batch = X[:,batch_size*(m//batch_size):]
            Y_batch = Y[:,batch_size*(m//batch_size):]
            minibatches.append((X_batch, Y_batch))
    else:
        minibatches.append((X,Y))
        batch_size = m
    
    # Set up placeholders for data
    n_L = 10
    X_train = tf.placeholder(tf.float32, shape=(n_0,None), name='X_train')
    Y_train = tf.placeholder(tf.int32, shape=(10,None), name='Y_train')
    global_step = tf.Variable(0, trainable=False)
    
    # Set up network
    n_l = (n_0, 60, 30, n_L)
    tensors = {'A0':X_train}
    regularizer = tf.constant(0.0)
    for l in range(1,len(n_l)):
        tensors['W'+str(l)] = tf.Variable(1.0*np.random.rand(n_l[l],n_l[l-1]), dtype=tf.float32, name='W'+str(l))
        tensors['b'+str(l)] = tf.Variable(0.1*np.ones((n_l[l],1)), dtype=tf.float32, name='b'+str(l))
        tensors['Z'+str(l)] = tf.add(tf.matmul(tensors['W'+str(l)],tensors['A'+str(l-1)]),tensors['b'+str(l)], name='Z'+str(l))
        if l != len(n_l)-1:
#            tensors['A'+str(l)] = tf.nn.relu(tensors['Z'+str(l)], name='A'+str(l))
            tensors['A'+str(l)] = tf.maximum(0.01*tensors['Z'+str(l)], tensors['Z'+str(l)], name='A'+str(l))
        else:
            tensors['A'+str(l)] = tf.nn.sigmoid(tensors['Z'+str(l)], name='A'+str(l))
        if lambd != 0:
            regularizer = tf.add(regularizer, tf.reduce_sum(tf.square(tensors['W'+str(l)])))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tensors['A'+str(len(n_l)-1)], labels=Y_train, dim=0))
    cost = loss + (lambd/m)*regularizer
    
    # Set up optimizer
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, len(minibatches), 10**(-1/num_epochs), staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.99, beta2=0.990)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    
    # Set up training loop
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        i = 0
        step, train_cost_y, val_cost_y = ([],[],[])
        plt.ion()
        batch_cost = sess.run(cost, feed_dict={X_train:X, Y_train:Y})
        val_cost = sess.run(cost, feed_dict={X_train:X_val, Y_train:Y_val})
        print('Progress: {}/{} ({:2}%), epoch: 0, batch: 0\n\tTraining cost: {:4}, validation cost: {:4}'.format(i, num_epochs*len(minibatches), 100*i/(num_epochs*len(minibatches)), batch_cost, val_cost))
        step.append(i)
        train_cost_y.append(batch_cost)
        val_cost_y.append(val_cost)
        plt.figure('MNIST training loss')
        plt.clf()
        plt.plot(step, train_cost_y, label='Training cost')
        plt.plot(step, val_cost_y, label='Validation cost')
        plt.legend()
        plt.ylim(ymin=0)
        plt.draw()
        plt.pause(1e-9)
        for epoch in range(num_epochs):
            for batch in range(len(minibatches)):
                i += 1
                _, batch_cost = sess.run([train_op, cost], feed_dict={X_train:minibatches[batch][0], Y_train:minibatches[batch][1]})
                if (i % (len(minibatches)//2) == 0):
                    val_cost = sess.run(cost, feed_dict={X_train:X_val, Y_train:Y_val})
                    print('Progress: {}/{} ({:.2%}), epoch: {}, batch: {}\n\tTraining cost: {:4}, validation cost: {:4}'.format(i, num_epochs*len(minibatches), i/(num_epochs*len(minibatches)), epoch, batch, batch_cost, val_cost))
                    step.append(i)
                    train_cost_y.append(batch_cost)
                    val_cost_y.append(val_cost)
                    plt.figure('MNIST training loss')
                    plt.clf()
                    plt.plot(step, train_cost_y, label='Training cost')
                    plt.plot(step, val_cost_y, label='Validation cost')
                    plt.legend()
                    plt.ylim(ymin=0)
                    plt.draw()
                    plt.pause(1e-9)
#        plt.ioff()
    
        # Evaluate prediction
        train_pred = sess.run(tensors['A'+str(len(n_l)-1)], feed_dict={X_train:X})
        train_accuracy = np.mean(np.all((train_pred>0.5)==Y, axis=0))
        if (X_val is not None) and (Y_val is not None):
            val_pred = sess.run(tensors['A'+str(len(n_l)-1)], feed_dict={X_train:X_val, Y_train:Y_val})
            val_accuracy = np.mean(np.all((val_pred>0.5)==Y_val, axis=0))
        print('Training accuracy: {}, validation accuracy: {}'.format(train_accuracy, val_accuracy))
    
        # Write test predictions to csv
        if (X_test is not None):
            print('Writing prediction of test set to CSV.')
            test_pred = sess.run(tensors['A'+str(len(n_l)-1)], feed_dict={X_train:X_test})
            test_pred_data = list(zip(1+np.arange(test_pred.shape[1]), np.argmax(test_pred, axis=0)))
#            test_pred_data = []
#            for col in range(test_pred.shape[1]):
#                for row in range(test_pred.shape[0]):
#                    if test_pred[row,col] > 0.5:
#                        test_pred_data.append((col,row))
#                    else:
#                        test_pred_d
#                        break
            test_pred_df = pd.DataFrame(data=test_pred_data, columns=['ImageId','Label'])
            test_pred_df.to_csv('MNIST_test_predictions.csv', index=False, header=True)
            print('Done!')
    
    # Return predictions
    return train_pred, val_pred





# Train the network, print out predictions (should get something like 95% accuracy on val set)
#train_pred, val_pred = NeuralNetwork(X_train, Y_train, init_learning_rate=0.03,
#                                     num_epochs=100, batch_size=1000, lambd=0.1,
#                                     X_val=X_val, Y_val=Y_val, X_test=X_test)

# Plot some classified samples from the test set
#Y_test_raw = pd.read_csv('./MNIST_test_predictions.csv')
#X_test_raw = pd.read_csv('./test.csv')
Y_test = np.array(Y_test_raw.iloc[:,1])
X_test = np.array(X_test_raw).T/255
print('Plotting samples')
plot_samples(X_test, Y=Y_test, idx=np.random.permutation(len(Y_test)))



















