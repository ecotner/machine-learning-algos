# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:33:56 2018

Building an RNN from scratch. Following the blog post https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

@author: Eric Cotner
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 2
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    ''' Generates a random binary string, then shifts all elements by <echo_step> to create target labels. Then organizes into batches. '''
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step).astype(int)
    y[:echo_step] = 0
    
    x = x.reshape([batch_size, -1])
    y = y.reshape([batch_size, -1])
    
    return (x,y)

# Set up computational graph
tf.reset_default_graph()
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

# Initial hidden state (probably zeros?)
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# Recurrent hidden layer parameters
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros([1, state_size]), dtype=tf.float32)

# Fully connected layer parameters
W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

# Unrolling minibatches
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass - unrolling RNN cells
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size,1])
    input_and_state_concatenated = tf.concat([current_input, current_state], axis=1)
    
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    states_series.append(next_state)
    current_state = next_state

# Append fully-connected layer to the end of unrolled RNN cells
logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series, labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

# Plots to monitor training
def plot(loss_list, predictions_series, batchX, batchY):
    # Plot the loss
    plt.subplot(2,3,1)
    plt.cla()
    plt.loglog(loss_list)
    
    # Plots the input and output as bar graphs?
    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        
        plt.subplot(2,3, batch_series_idx+2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0 ,2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series *0.3, width=1, color="green")
    
    plt.draw()
    plt.pause(1e-6)

# Run the TF session and perform training
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    
    for epoch_idx in range(num_epochs):
        x, y = generateData()
        _current_state = np.zeros((batch_size, state_size)) # The initial state at each epoch
        
        print('New data, epoch', epoch_idx)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            
            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]
            
            _total_loss, _train_step, _current_state, _predictions_series = sess.run([total_loss, train_step, current_state, predictions_series], feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state})
            
            loss_list.append(_total_loss)
            
            if batch_idx%100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()






















