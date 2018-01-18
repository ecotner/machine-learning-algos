# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:17:47 2018

Runs inference on the prediction of the classification network.

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import utils as u
import time
import re

# Define parameters
CIFAR_DATA_PATH = './CIFAR10_data/cifar-10-batches-py/'
SAVE_PATH = './checkpoints/{0}/CIFAR10_{0}'.format(11)
N_DROPOUT_GROUPS = 3
PREDICTION_PERIOD = 2.5 # Number of second between displaying successive predictions
BATCH_SIZE = 256

# Determine whether to run inference on entire set or batches of images at a time
x = u.input_choice('Run inference on images one by one?\n[y/n]: ', choices=['y','n'])
if x == 'y':
    RUN_INDIVIDUAL_INFERENCE = True
else:
    RUN_INDIVIDUAL_INFERENCE = False

# Load dataset
print('Loading test data...')
data = u.unpickle(CIFAR_DATA_PATH+'test_batch')
label_names = u.unpickle(CIFAR_DATA_PATH+'batches.meta')[b'label_names']
label_names = [b_str.decode() for b_str in label_names]

# (Pre-)process data
print('Processing data...')
X_test = np.transpose(np.reshape(data[b'data'], [-1,32,32,3], order='F'), axes=[0,2,1,3])
Y_test = data[b'labels']
# Get mean and std from log file
p_mean = re.compile('X_mean')
p_std = re.compile('X_std')
p_float = re.compile(r'\d+\.\d+')
with open(SAVE_PATH+'.log', 'r') as fo:
    for line in fo:
        if re.match(p_mean, line) is not None:
            m = re.search(p_float, line)
            X_mean = float(m.group())
        elif re.match(p_std, line) is not None:
            m = re.search(p_float, line)
            X_std = float(m.group())
# Apply normalization and permutation
X_test = (X_test - X_mean)/X_std
seed = int(time.time())
np.random.seed(seed)
X_test = np.random.permutation(X_test)
np.random.seed(seed)
Y_test = np.random.permutation(Y_test)

# Load tensorflow graph
G = tf.Graph()
with G.as_default():
    print('Loading graph...')
    tf.device('/cpu:0')
    saver = tf.train.import_meta_graph(SAVE_PATH+'.meta', clear_devices=True)
    tf.device('/cpu:0')
    
    # Get important tensors from the graph
    X = G.get_tensor_by_name('input:0')
    Y = G.get_tensor_by_name('output:0')
    labels = G.get_tensor_by_name('labels:0')
    J = G.get_tensor_by_name('loss:0')
    regularization_parameter = G.get_tensor_by_name('regularization_parameter:0')
    keep_prob = {n:G.get_tensor_by_name('keep_prob_'+str(n)+':0') for n in range(1,N_DROPOUT_GROUPS+1)}
    is_training = G.get_tensor_by_name('is_training:0')
    
    # Make any new necessary tensors:
    prob = tf.nn.softmax(Y) # Predicted probability distribution of classes
    
    # Begin tensorflow session
    sess_config = tf.ConfigProto(device_count={'CPU':4, 'GPU':0})
    with tf.Session(config=sess_config) as sess:
        print('Restoring parameters...')
        saver.restore(sess, SAVE_PATH)
        
        plt.figure('CIFAR-10 Prediction')
        
        print('Running prediction...')
        classification_score = {'right':0, 'wrong':0}
        for b in range(X_test.shape[0]//BATCH_SIZE+1):
            # Run prediction on images (do a large batch at a time so batch norm can get statistics, since it looks like the inference mode is broken)
            slice_lower = b*BATCH_SIZE
            slice_upper = min(X_test.shape[0], (b+1)*BATCH_SIZE)
            feed_dict = {**{X:X_test[slice_lower:slice_upper,:,:,:], labels:Y_test[slice_lower:slice_upper], is_training:True}, **{keep_prob[n]:1 for n in range(1,N_DROPOUT_GROUPS+1)}}
            class_prob, loss = sess.run([prob, J], feed_dict=feed_dict)
            class_pred = np.argmax(class_prob, axis=-1)
            if RUN_INDIVIDUAL_INFERENCE:
                # Plot each individual image in the batch
                for idx, lbl in enumerate(class_pred):
                    tic = time.time()
                    if lbl == Y_test[slice_lower+idx]:
                        right_or_wrong = '\u2714'
                    else:
                        right_or_wrong = 'X'
                    print('Class prob. dist: {}\nClass_pred: {}, loss: {:.2e}\n\n'.format(class_prob[idx], label_names[lbl], loss))
                    
                    # Plot the results
                    plt.clf()
                    img = X_test[slice_lower+idx,:,:,:]
                    img = (img - np.min(img))/(np.max(img) - np.min(img))
                    plt.imshow(img)
                    plt.title('Class: {}, prediction: {} ({:.1f}%) {}'.format(label_names[Y_test[slice_lower+idx]], label_names[lbl], 100*class_prob[idx,lbl], right_or_wrong))
                    plt.xticks([])
                    plt.yticks([])
                    plt.draw()
                    plt.pause(1e-9)
                    
                    # Time delay between plotting
                    toc = time.time()
                    time.sleep(max(0, PREDICTION_PERIOD - (toc-tic)))
            else:
                # Gather summary statistics
                print('Running batch {}/{}...'.format(b, X_test.shape[0]//BATCH_SIZE))
                n_right = np.sum(np.equal(class_pred, Y_test[slice_lower:slice_upper]))
                classification_score['right'] += n_right
                classification_score['wrong'] += (len(class_pred) - n_right)
        # Print summary stats
        if not RUN_INDIVIDUAL_INFERENCE:
            error = classification_score['wrong']/(classification_score['right'] + classification_score['wrong'])
            print('Test set error: {}\nNum correctly classified: {}, incorrectly classified: {}'.format(error, classification_score['right'], classification_score['wrong']))