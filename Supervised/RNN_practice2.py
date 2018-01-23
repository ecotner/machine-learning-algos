# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:33:57 2018

Now that I know how a basic RNN works, I'm going to try and get a network to reproduce a sine wave with an LSTM built from scratch, but doing the proper unrolling procedure for backprop.

@author: Eric Cotner
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

backprop_len = 10
LSTM_width = 25
dense_width = 25
batch_size = 20
seq_len = 200*backprop_len
noise_mag = 0.1

LEARNING_RATE = 1e-2
MAX_EPOCHS = 20

def generateData(batch_size, seq_len, period=10, noise_mag=0):
    ''' Generates batches of sine waves initialized with random phases (and possibly noise), and corresponding training targets. '''
    # Generate random phase between 0 and pi/2
    random_phase = 2*np.pi*np.random.rand(batch_size, 1)
    # Generate sine wave
    x = np.sin((2*np.pi/period) * np.tile(np.arange(seq_len+1), (batch_size,1)) + random_phase)
    # Shift wave by one increment
    y = np.roll(x, -1, axis=-1)
    # Add noise (if desired)
    x = x + noise_mag * np.random.standard_normal(x.shape)
    # Drop the last element of y in the series since it is not continuous with the first, and the first element of x to maintain the same length
    y = y[:,:-1]
    x = x[:,:-1]
    
    return x, y

# Set up computational graph
tf.reset_default_graph()

# Define placeholders for initial states and input/labels
X = tf.placeholder(dtype=tf.float32, shape=[None, backprop_len])
labels = tf.placeholder(dtype=tf.float32, shape=[None, backprop_len])
s0 = tf.placeholder_with_default(tf.zeros([tf.shape(X)[0], LSTM_width]), [X.shape[0], LSTM_width], name='init_state')
h0 = tf.placeholder_with_default(tf.zeros([tf.shape(X)[0], LSTM_width]), [X.shape[0], LSTM_width], name='init_state')

# Set up LSTM weights (U is associated with input, W with hidden state)
Ui = tf.Variable(tf.random_normal([1, LSTM_width]))
Wi = tf.Variable(tf.random_normal([LSTM_width, LSTM_width]))
bi = tf.Variable(tf.ones([LSTM_width]))

Uf = tf.Variable(tf.random_normal([1, LSTM_width]))
Wf = tf.Variable(tf.random_normal([LSTM_width, LSTM_width]))
bf = tf.Variable(tf.ones([LSTM_width]))

Uo = tf.Variable(tf.random_normal([1, LSTM_width]))
Wo = tf.Variable(tf.random_normal([LSTM_width, LSTM_width]))
bo = tf.Variable(tf.ones([LSTM_width]))

Us = tf.Variable(tf.random_normal([1, LSTM_width]))
Ws = tf.Variable(tf.random_normal([LSTM_width, LSTM_width]))
bs = tf.Variable(tf.ones([LSTM_width]))

# Compute gates and hidden states, unroll layer
s, h = s0, h0
LSTM_output_list = []
for t in range(backprop_len):
    i = tf.sigmoid(tf.matmul(tf.slice(X, [0,t], [-1,1]), Ui) + tf.matmul(h, Wi) + bi)
    f = tf.sigmoid(tf.matmul(tf.slice(X, [0,t], [-1,1]), Uf) + tf.matmul(h, Wf) + bf)
    o = tf.sigmoid(tf.matmul(tf.slice(X, [0,t], [-1,1]), Uo) + tf.matmul(h, Wo) + bo)
    s = f*s + i * tf.sigmoid(tf.matmul(tf.slice(X, [0,t], [-1,1]), Us) + tf.matmul(h, Ws) + bs)
    h = o * tf.tanh(s)
    LSTM_output_list.append(h)

# Apply dense layer to top
W2 = tf.Variable(tf.random_normal([LSTM_width, dense_width]))
b2 = tf.Variable(tf.zeros([dense_width]))
A2_list = [tf.nn.relu(tf.add(tf.matmul(h, W2), b2)) for h in LSTM_output_list]

# Output layer
W3 = tf.Variable(tf.random_normal([dense_width, 1]))
b3 = tf.Variable(tf.zeros([1]))
Y_list = [tf.add(tf.matmul(A2, W3), b3) for A2 in A2_list]

# Calculate loss, create training operation
Y = tf.concat(Y_list, axis=-1)
losses = tf.square(Y - labels)
loss = tf.reduce_mean(losses)

lr = tf.placeholder(dtype=tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)

# Execute training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.figure('loss')
    plt.title('Loss')
    plt.xlabel('Global step')
    plt.ylabel('Loss')
    global_step_list = []
    loss_list = []
    plt.ion()
    
    # Run loop over epochs
    global_step = 0
    for epoch in range(MAX_EPOCHS):
        x, y = generateData(batch_size, seq_len, noise_mag=noise_mag)
        s_, h_ = np.zeros([batch_size, LSTM_width]), np.zeros([batch_size, LSTM_width])
        
        # Loop over batches
        for b in range(seq_len//backprop_len):
            start_idx = b*backprop_len
            end_idx = start_idx + backprop_len
            feed_dict = {X:x[:,start_idx:end_idx], labels:y[:,start_idx:end_idx], s0:s_, h0:h_, lr:LEARNING_RATE}
            loss_, s_, h_, _ = sess.run([loss, s, h, train_op], feed_dict=feed_dict)
            
            print('Epoch: {}, batch: {}/{}, log10(loss): {}'.format(epoch+1, b+1, seq_len//backprop_len, np.log10(loss_)))
            
            if (global_step % 10 == 0):
                global_step_list.append(global_step)
                loss_list.append(loss_)
                plt.cla()
                plt.semilogy(global_step_list, loss_list, 'o', alpha=0.25)
                plt.draw()
                plt.pause(1e-6)
            
            # Iterate global step
            global_step += 1
    
    # Run prediction (ie compare output with expected)
    plt.figure('Prediction')
    plt.cla()
    seq_len = 4*backprop_len
    x, y = generateData(1, seq_len)
    s_, h_ = np.zeros([1, LSTM_width]), np.zeros([1, LSTM_width])
    predictions_list = []
    
    for b in range(seq_len//backprop_len):
        start_idx = b*backprop_len
        end_idx = start_idx + backprop_len
        feed_dict = {X:x[:,start_idx:end_idx], s0:s_, h0:h_}
        predictions, s_, h_ = sess.run([Y_list, s, h], feed_dict=feed_dict)
        for p in predictions:
            predictions_list.append(float(p.squeeze()))
    
    # Plot the outcome
    plt.plot(x.squeeze(), label='Input')
    plt.plot(y.squeeze(), label='Expected')
    plt.plot(predictions_list, label='Predicted')
    plt.legend()
    plt.draw()
    plt.pause(1e-6)

plt.ioff()
plt.show()












