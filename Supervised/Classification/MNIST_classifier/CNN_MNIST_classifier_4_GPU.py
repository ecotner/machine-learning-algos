# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:41:12 2017

Convolutional neural network for classifying MNIST handwritten digits.

TO DO: add a third convolutional layer - try to see what happens when removing
the pool layers to get more resolution?

@author: Eric Cotner
"""

# Import necessary modules
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Make sure current working directory is same as file directory
#file_dir = os.path.dirname(__file__)
#os.chdir(file_dir)

# Enable logging
tf.logging.set_verbosity(tf.logging.INFO)

''' ===================== IMPORT AND PREPROCESS DATASET ==================== '''

def get_data():
    print('Loading dataset')
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
    Y_train = Y_one_hot[:(m-750),:]
    X_train = X[:(m-750),:]
    Y_val = Y_one_hot[(m-750):,:]
    X_val = X[(m-750):,:]
    return X_train, Y_train, X_val, Y_val

#X_train, Y_train, X_val, Y_val = get_data()
    
def make_batches(X, Y, batch_size):
    ''' Makes minibatches.
    Args:
        X: Design matrix; assumes dimensions [num_examples, num_features]
        Y: One-hot label matrix; assumes dimensions [num_examples, num_classes]
        batch_size: the mini-batch size
    Output:
        batches: list of tuples (X_batch, Y_batch) containing the minibatches
    '''
    num_examples = X.shape[0]
    assert num_examples == Y.shape[0], 'X and Y don\'t have same first dimension'
    num_full_batches = num_examples//batch_size
    batches = []
    for k in range(num_full_batches):
        X_batch = X[k*batch_size:(k+1)*batch_size,:]
        Y_batch = Y[k*batch_size:(k+1)*batch_size,:]
        batches.append((X_batch, Y_batch))
    if num_examples % num_full_batches != 0:
        X_batch = X[num_full_batches*batch_size-1:,:]
        Y_batch = Y[num_full_batches*batch_size-1:,:]
        batches.append((X_batch, Y_batch))
    return batches

# Show some samples from the data
def plot_samples(X, Y=None, idx=None):
    ''' Plots the digits, with labels Y if given. '''
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

''' =================== CREATE THE COMPUTATION GRAPH ======================= '''

# Reset the graph
tf.reset_default_graph()

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
def lrelu(x):
    return tf.maximum(0.01*x, x)

# Raw input X and one-hot labels Y
X = tf.placeholder(tf.float32, shape=[None, 28*28])
Y = tf.placeholder(tf.int32, shape=[None, 10])
# Other placeholders for hyperparameters
learning_rate_tensor = tf.placeholder(tf.float32, shape=[])
keep_prob_tensor = tf.placeholder(tf.float32)

# Input layer, reshaped to 2D image
input_layer = tf.reshape(tf.cast(X, dtype=tf.float32), [-1,28,28,1])
    
# Define first convolution layer
W_conv1 = weight_variable([5,5,1,32], 'W_conv1') # 32 5x5x1 filters
b_conv1 = bias_variable([32], 'b_conv1')
conv1 = lrelu(conv2d(input_layer, W_conv1) + b_conv1)
pool1 = max_pool_2x2(conv1) # pool1 layer has shape 14x14x32

# Define second convolution layer
W_conv2 = weight_variable([5,5,32,64], 'W_conv2') # 64 5x5x32 filters
b_conv2 = bias_variable([64], 'b_conv2')
conv2 = lrelu(conv2d(pool1, W_conv2) + b_conv2)
pool2 = max_pool_2x2(conv2) # Pool2 layer has shape 7x7x64

# Flatten pool layer and feed into dense layer
pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
W_dense3 = weight_variable([7*7*64, 1024], 'W_dense3')
b_dense3 = bias_variable([1024], 'b_dense3')
dense3 = lrelu(tf.matmul(pool2_flat, W_dense3) + b_dense3)

# Apply dropout before output layer
dropout4 = tf.nn.dropout(dense3, keep_prob_tensor)

# Final output layer
W_out5 = weight_variable([1024, 10], 'W_out5')
b_out5 = bias_variable([10], 'b_out5')
y_out = tf.matmul(dropout4, W_out5) + b_out5 # This is NOT the class prediction; have to take softmax later

xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=Y, dim=1))
probabilities = tf.nn.softmax(y_out, dim=1)
prediction = tf.argmax(probabilities, axis=1)
optimizer = tf.train.AdamOptimizer(learning_rate_tensor)
train_op = optimizer.minimize(xentropy, global_step=tf.train.get_global_step())

''' ========================= TRAIN THE NETWORK =========================== '''

def accuracy(Y, Y_pred):
    ''' Takes in two one-hot label matrices and gives back the accuracy. '''
    Y_label = np.argmax(Y, axis=1)
    Y_pred_label = np.argmax(Y_pred, axis=1)
    return np.mean(Y_label == Y_pred_label)

def train(X_train, Y_train, X_val, Y_val, learning_rate, max_epochs, 
          batch_size=128, keep_prob=0.5, plot_every_n_steps=np.inf, recover=True):
    ''' Initiates the training of the network. '''
    
    # Split training set into minibatches
    train_batches = make_batches(X_train, Y_train, batch_size)
    # Initialize stuff for plotting learning curves
    plt.ion()
    step_list = []
    train_loss_list = []
    train_error_list = []
    val_loss_list = []
    val_accuracy_list = []
    # Initialize saver
    saver = tf.train.Saver(var_list=None, max_to_keep=5, keep_checkpoint_every_n_hours=1)
    # Initialize GPU options
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    # Start TF session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # Load/save meta graph and initial variables
        if recover == True:
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
        else:
            saver.save(sess, save_path='./checkpoints/MNIST_CNN_v2')
        # Iterate over each epoch
        print('Beginning training')
        for epoch in range(max_epochs):
            # Iterate over each minibatch
            for step, batch in enumerate(train_batches):
#                X_batch, Y_batch = batch
                # Do the training step
                _, train_loss = sess.run([train_op, xentropy], feed_dict={X:batch[0], Y:batch[1],
                         learning_rate_tensor:learning_rate,
                         keep_prob_tensor:keep_prob})
                print('Epoch: {}, batch: {}, training loss: {}'.format(epoch, step, train_loss))
                # Save variables periodically
                if (epoch*len(train_batches)+step) % 25 == 0:
                    print('Saving checkpoint')
                    saver.save(sess, save_path='./checkpoints/MNIST_CNN_v2', 
                               global_step=1, write_meta_graph=False)
                # Plot loss and accuracy
                if (epoch*len(train_batches)+step) % plot_every_n_steps == 0:
                    # Get subset of training set which is same size as val set
#                    perm = np.random.permutation(X_train.shape[0])[:X_val.shape[0]]
#                    X_batch = X_train[perm,:]
#                    Y_batch = Y_train[perm,:]
                    # Get training/validation loss/predictions
                    train_loss, train_prob = sess.run([xentropy, probabilities],
                                                      feed_dict={X:batch[0], Y:batch[1],
                                                                 keep_prob_tensor:1.0})
                    val_loss, val_prob = sess.run([xentropy, probabilities],
                                                  feed_dict={X:X_val, Y:Y_val,
                                                             keep_prob_tensor:1.0})
                    # Append new values to each list
                    step_list.append(epoch+step/len(train_batches))
                    train_loss_list.append(train_loss)
                    train_error_list.append(1-accuracy(batch[1], train_prob))
                    val_loss_list.append(val_loss)
                    val_accuracy_list.append(1-accuracy(Y_val, val_prob))
                    # Plot the loss
                    plt.figure('loss')
                    plt.clf()
                    plt.semilogy(step_list, train_loss_list, label='Training')
                    plt.semilogy(step_list, val_loss_list, label='Validation')
                    plt.title('Cross-entropy loss vs. epoch')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.draw()
                    plt.pause(1e-10)
                    # Plot the accuracy
                    plt.figure('error')
                    plt.clf()
                    plt.semilogy(step_list, train_error_list, label='Training')
                    plt.semilogy(step_list, val_accuracy_list, label='Validation')
                    plt.title('Prediction error (1-accuracy) vs. epoch')
                    plt.xlabel('Epoch')
                    plt.ylabel('Error rate')
                    plt.legend()
                    plt.draw()
                    plt.pause(1e-10)

# Load training/validation data
#X_train, Y_train, X_val, Y_val = get_data()
# Train the network
#train(X_train, Y_train, X_val, Y_val, learning_rate=3e-5, max_epochs=100, 
#      batch_size=1000, plot_every_n_steps=10, recover=True)

''' ========================= MAKE PREDICTIONS ============================= '''

def predict():
    # Load and process test data
    print('Loading test data')
    test_raw = pd.read_csv('./test.csv')
    X_test = np.array(test_raw)/255
    
    # Reload data from last saved checkpoint during training
    saver = tf.train.Saver(var_list=None)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Restoring parameters')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
    
        # Make predictions on test data
        # Probably want to break this up into batches so it's not running out of
        # memory trying to do all the matrix multiplications
        print('Calculating predictions')
        batches = make_batches(X_, np.zeros((X_.shape[0],1)), batch_size=500)
        test_pred_list = []
        for batch in batches:
            temp_pred = sess.run(prediction, feed_dict={X:batch[0], keep_prob_tensor:1.0})
            test_pred_list.append(temp_pred)
        test_pred = np.concatenate(test_pred_list, axis=0)
        
        # Save predictions to CSV
        print('Saving predictions to CSV')
        test_pred = np.concatenate((np.reshape(np.arange(test_pred.shape[0])+1, (-1,1)),
                                    np.reshape(test_pred, (-1,1))), axis=1)
        test_pred_df = pd.DataFrame(data=test_pred, columns=['ImageId','Label'])
        test_pred_df.to_csv('./MNIST_CNN_test_predictions.csv', index=False)

#predict()

''' ========================== VISUALIZE STUFF ========================== '''

# Load and process test data
def visualize_test():
    print('Loading test data')
    test_raw = pd.read_csv('./test.csv')
    test_pred_raw = pd.read_csv('./MNIST_CNN_test_predictions.csv')
    X_test = np.array(test_raw)
    Y_test_label = np.array(test_pred_raw.iloc[:,1])
    Y_test = np.zeros((Y_test_label.shape[0], 10))
    for idx, label in enumerate(Y_test_label):
        Y_test[idx,label] = 1
    plot_samples(X_test, Y_test_label, np.random.permutation(Y_test_label.shape[0]))

# Visualize misclassified training/validation data
def visualize_misclassification():
    print('Loading training data')
    train_raw = pd.read_csv('./train.csv')
    X_ = np.array(train_raw.iloc[:,1:])/255
    Y_ = np.array(train_raw.iloc[:,0])
    
    # Get predictions
    saver = tf.train.Saver(var_list=None)
    # Configure GPU settings
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
#        sess.run(tf.global_variables_initializer())
        print('Restoring parameters')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
    
        # Make predictions on test data
        print('Calculating predictions')
        batches = make_batches(X_, np.zeros((X_.shape[0],1)), batch_size=500)
        test_pred_list = []
        for batch in batches:
            temp_pred = sess.run(prediction, feed_dict={X:batch[0], keep_prob_tensor:1.0})
            test_pred_list.append(temp_pred)
        test_pred = np.concatenate(test_pred_list, axis=0)
        
        # Identify misclassified data
        misclassified_idx = np.where(Y_ != test_pred)
        X_wrong = X_[misclassified_idx]
        Y_wrong = Y_[misclassified_idx]
        test_pred_wrong = test_pred[misclassified_idx]
        plt.figure('MNIST Misclassification')
        plt.ion()
        break_flag = False
        while True:
            for i in range(X_wrong.shape[0]):
                plt.clf()
                plt.imshow(X_wrong[i].reshape((28,28)))
                plt.title('Prediction: {}, actual: {}'.format(test_pred_wrong[i], Y_wrong[i]))
                plt.draw()
                plt.pause(1e-10)
                q = input('Press any key for next image, or q to quit >')
                if q.lower() == 'q':
                    break_flag = True
                    break
            if break_flag == True:
                break
            
#visualize_test()
visualize_misclassification()












