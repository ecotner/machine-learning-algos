# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 13:22:53 2018

Building a recurrent neural network (RNN) form scratch. One primary goal is to write in such a way that the hidden state is handled internally (rather than reading it out from the previous step and feeding it back in through feed_dict).

The task to be learned will be to simply predict the shape of a sine wave based on looking at the previous points.

@author: Eric Cotner
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define seeds for reproducibility
seed=1
np.random.seed(seed)

# Define basic RNN cell with tanh'd output
def RNNcell(X, n_in, num_units):
#    n_in = X.shape.as_list()[-1]
    
    # Set up the weights/bias
    Wx = tf.Variable(tf.random_normal(shape=[n_in, num_units], dtype=tf.float32, seed=seed)*tf.sqrt(2/tf.cast(n_in+num_units, tf.float32)))
    Wh = tf.Variable(tf.random_normal(shape=[num_units, num_units], dtype=tf.float32, seed=seed)*tf.sqrt(1/tf.cast(num_units, tf.float32)))
    b = tf.Variable(tf.ones(shape=[num_units]), dtype=tf.float32)
    
    # Initial hidden state is just a vector of all zeros
    h0 = tf.Variable(tf.zeros(shape=[tf.shape(X)[0], num_units], dtype=tf.float32), trainable=False, validate_shape=False)
    h = tf.assign(h0, tf.tanh(tf.matmul(X, Wx) + tf.matmul(h0, Wh) + b), validate_shape=False)
    return h, [h0]

def LSTMcell(X, n_in, num_units):
#    n_in = X.shape.as_list()[-1]
    
    # Set up the weights/biases
    U = tf.Variable(tf.random_normal(shape=[n_in, num_units], dtype=tf.float32)*tf.sqrt(2/tf.cast(n_in+num_units, tf.float32)))
    W = tf.Variable(tf.random_normal(shape=[num_units, num_units], dtype=tf.float32)*tf.sqrt(1/tf.cast(num_units, tf.float32)))
    b = tf.Variable(tf.ones(shape=[num_units]), dtype=tf.float32)
    Ui = tf.Variable(tf.random_normal(shape=[n_in, num_units], dtype=tf.float32)*tf.sqrt(2/tf.cast(n_in+num_units, tf.float32)))
    Wi = tf.Variable(tf.random_normal(shape=[num_units, num_units], dtype=tf.float32)*tf.sqrt(1/tf.cast(num_units, tf.float32)))
    bi = tf.Variable(tf.ones(shape=[num_units]), dtype=tf.float32)
    Uf = tf.Variable(tf.random_normal(shape=[n_in, num_units], dtype=tf.float32)*tf.sqrt(2/tf.cast(n_in+num_units, tf.float32)))
    Wf = tf.Variable(tf.random_normal(shape=[num_units, num_units], dtype=tf.float32)*tf.sqrt(1/tf.cast(num_units, tf.float32)))
    bf = tf.Variable(tf.ones(shape=[num_units]), dtype=tf.float32)
    Uo = tf.Variable(tf.random_normal(shape=[n_in, num_units], dtype=tf.float32)*tf.sqrt(2/tf.cast(n_in+num_units, tf.float32)))
    Wo = tf.Variable(tf.random_normal(shape=[num_units, num_units], dtype=tf.float32)*tf.sqrt(1/tf.cast(num_units, tf.float32)))
    bo = tf.Variable(tf.ones(shape=[num_units]), dtype=tf.float32)
    
    # Initialize hidden state/vector
    h0 = tf.Variable(tf.zeros(shape=[tf.shape(X)[0], num_units], dtype=tf.float32), trainable=False, validate_shape=False)
    s0 = tf.Variable(tf.zeros(shape=[tf.shape(X)[0], num_units], dtype=tf.float32), trainable=False, validate_shape=False, name='derp')
    
    # Calculate the gate variables
    i = tf.sigmoid(tf.matmul(X, Ui) + tf.matmul(h0, Wi) + bi)
    f = tf.sigmoid(tf.matmul(X, Uf) + tf.matmul(h0, Wf) + bf)
    o = tf.sigmoid(tf.matmul(X, Uo) + tf.matmul(h0, Wo) + bo)
    
    # Calculate internal state and hidden vector
    s = tf.assign(s0, f*s0 + i*tf.sigmoid(b + tf.matmul(X,U) + tf.matmul(h0,W)), validate_shape=False, name='state')
    h = tf.assign(h0, tf.tanh(s)*o, validate_shape=False, name='hidden')
    return h, [h0, s0]

tf.reset_default_graph()

# Create RNN architecture
lookback_len = 5
batch_size = 10
output_size = 1
hidden_size = 50
period = 10*lookback_len
X = tf.placeholder(dtype=tf.float32, shape=[None,1])
A, reset_tensors = LSTMcell(X, 1, hidden_size)
W = tf.Variable(tf.random_normal(shape=[hidden_size, output_size], dtype=tf.float32, seed=seed))
Y = tf.matmul(A, W)

# Create training loss and operation
learning_rate = 1e-2
labels = tf.placeholder(dtype=tf.float32, shape=[batch_size, output_size])
loss = tf.reduce_mean(tf.square(Y-labels))
opt = tf.train.GradientDescentOptimizer(learning_rate)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    tf.set_random_seed(seed)
    sess.run(tf.global_variables_initializer(), feed_dict={X:np.zeros([batch_size, 1])})
    
    # Train
    num_epochs = 10**3
    for j in range(num_epochs):
        sess.run(tf.variables_initializer(reset_tensors), feed_dict={X:np.zeros([batch_size,1])})
        random_phase = np.random.randn(batch_size,1)
        y = np.sin(random_phase)
        for i in range(lookback_len):
            if np.random.rand() < 0.5:
                feed_dict = {X:np.sin((2*np.pi/period)*i + random_phase), labels:np.sin((2*np.pi/period)*(i+1) + random_phase)}
            else:
                feed_dict = {X:y, labels:np.sin((2*np.pi/period)*(i+1) + random_phase)}
            y, J, _ = sess.run([Y, loss, train_op], feed_dict=feed_dict)
        print('Epoch: {}, log10(J): {}'.format(j, np.log10(J)))
    
    # Test
    random_phase = np.random.randn(1,1)
    sess.run(tf.variables_initializer(reset_tensors), feed_dict={X:np.zeros([1,1])})
    y = np.sin(random_phase)
    predictions = []
    pred_len = 3*period
    for i in range(pred_len):
#        y = sess.run(Y, feed_dict={X:y})
        y = sess.run(Y, feed_dict={X:np.sin((2*np.pi/period)*i+random_phase)})
        predictions.append(y.squeeze())
    
    plt.figure('pred')
    plt.clf()
    plt.plot((2*np.pi/period)*(np.arange(pred_len)+1) + random_phase.squeeze(), predictions, label='Predictions')
    plt.plot((2*np.pi/period)*(np.arange(pred_len)+1) + random_phase.squeeze(), np.sin((2*np.pi/period)*(np.arange(pred_len)) + random_phase.squeeze()), label='sin(x)')
    plt.legend()
















