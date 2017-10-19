# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:48:14 2017

Simple neural network to classify Gaussian-distributed 2D data by which quadrant
each data point is in.

@author: 27182_000
"""

import tensorflow as tf
import numpy as np

# Set up NN architecture
tf.reset_default_graph()
n0, n1, n2 = (2, 10, 4)

# Data
X = tf.placeholder(tf.float32, shape=[n0,None], name='X')
Y = tf.placeholder(tf.float32, shape=[n2,None], name= 'Y')

# Parameter initialization
W1 = tf.Variable(0.01*np.random.randn(n1,n0), dtype=tf.float32, name='W1')
b1 = tf.Variable(np.zeros((n1,1)), dtype=tf.float32, name='b1')
W2 = tf.Variable(0.01*np.random.randn(n2,n1), dtype=tf.float32, name='W2')
b2 = tf.Variable(np.zeros((n2,1)), dtype=tf.float32, name='b2')

# Set up computational graph
Z1 = tf.add(tf.matmul(W1,X),b1, name='Z1')
A1 = tf.nn.sigmoid(Z1, name='A1')
Z2 = tf.add(tf.matmul(W2,A1),b2, name='Z2')
A2 = tf.nn.softmax(Z2, dim=0, name='A2')

# Set up cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=A2, labels=Y, dim=0), name='cost')

#with tf.Session() as sess:
#    init_op = tf.global_variables_initializer()
#    sess.run(init_op)
#    # Visualize graph structure
#    writer = tf.summary.FileWriter('logs', sess.graph)
#    writer.close()

# Set up optimizer
learning_rate = 0.03
momentum = 0.9
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op= optimizer.minimize(cost)

# Create data
m = 1000
X_data = np.random.randn(2,m)
Y_data = np.zeros((4,m))
for i in range(m):
    if X_data[0,i] > 0 and X_data[1,i] > 0:
        Y_data[0,i] = 1
    elif X_data[0,i] < 0 and X_data[1,i] > 0:
        Y_data[1,i] = 1
    elif X_data[0,i] < 0 and X_data[1,i] < 0:
        Y_data[2,i] = 1
    elif X_data[0,i] > 0 and X_data[1,i] < 0:
        Y_data[3,i] = 1
X_train = X_data[:,:900]
Y_train = Y_data[:,:900]
X_val = X_data[:,900:]
Y_val = Y_data[:,900:]

# Run training loop
num_epochs = 10000
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for step in range(num_epochs):
        _, batch_cost = sess.run([train_op, cost], feed_dict={X:X_train, Y:Y_train})
        if step%100==0:
            print('Step {}: loss={:3f}'.format(step, batch_cost))
    
# Evaluate prediction
    pred = sess.run(A2, feed_dict={X:X_val, Y:Y_val})
    accuracy = np.mean(np.all((pred>0.5)==Y_val, axis=0))
    print('Accuracy: {}'.format(accuracy))



































