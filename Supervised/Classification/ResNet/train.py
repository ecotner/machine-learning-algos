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
LEARNING_RATE_ANNEAL_STEP = int(1e9)     # Number of epochs after which learning rate is annealed by 1/e
REGULARIZATION_TYPE = 'L2'  # Regularization type is already determined in ResNet.py
REGULARIZATION_PARAMETER = 3e-1
INPUT_NOISE_MAGNITUDE = np.sqrt(0.10)
KEEP_PROB = {1: 1, 2: 0.6, 3: 0.7}
VAL_KEEP_PROB = {1:1, 2:1, 3: 1}
SAVE_PATH = './checkpoints/{0}/CIFAR10_{0}'.format(8)
MAX_EPOCHS = int(1e10)
CIFAR_DATA_PATH = './CIFAR10_data/cifar-10-batches-py/'
LOG_EVERY_N_STEPS = 100
GPU_MEM_FRACTION = 0.6

# Load image data. Returns a dict that contains 'data' (10000x3072 numpy array of uint8's) and 'labels' (list of 10000 numbers between 0-9 denoting the class). There are 5 training data files, one test file, and one 'meta' file containing a list of human-decipherable names to the classes.
print('Loading data...')
data = []
for n in range(1,5+1):
    data.append(u.unpickle(CIFAR_DATA_PATH+'data_batch_'+str(n)))
data.append(u.unpickle(CIFAR_DATA_PATH+'test_batch'))
data.append(u.unpickle(CIFAR_DATA_PATH+'batches.meta'))

# Extract data from dict
print('Processing data...')
X = np.concatenate([data[n][b'data'] for n in range(5)], axis=0).reshape([-1,32,32,3])
Y = np.concatenate([data[n][b'labels'] for n in range(5)], axis=0)
X_test = data[5][b'data'].reshape([-1,32,32,3])
Y_test = np.array(data[5][b'labels'])
label_names = data[6]
del data

# Preprocess data by normalizing to zero mean and unit variance
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean)/X_std
X_test = (X_test - X_mean)/X_std

# Shuffle data and split into training and validation sets
np.random.seed(0)
X_train = np.random.permutation(X)
np.random.seed(0)
Y_train = np.random.permutation(Y)
del X, Y
m_train = len(Y_train) - VAL_BATCH_SIZE
m_test = len(Y_test)

# Log some info about the data for future use
with open(SAVE_PATH+'.log', 'w+') as fo:
    fo.write('Training log\n\n')
    fo.write('Dataset metrics:\n')
    fo.write('Training data shape: {}\n'.format(X_train.shape))
    fo.write('Validation set size: {}\n'.format(VAL_BATCH_SIZE))
    fo.write('Test data size: {}\n'.format(X_test.shape[0]))
    fo.write('X_mean: {}\n'.format(X_mean))
    fo.write('X_std: {}\n\n'.format(X_std))
    fo.write('Hyperparameters:\n')
    fo.write('Batch size: {}\n'.format(BATCH_SIZE))
    fo.write('Learning rate: {}\n'.format(LEARNING_RATE))
    fo.write('Learning rate annealed every N epochs: {}\n'.format(LEARNING_RATE_ANNEAL_STEP))
    fo.write('Regularization type: {}\n'.format(REGULARIZATION_TYPE))
    fo.write('Regularization parameter: {}\n'.format(REGULARIZATION_PARAMETER))
    fo.write('Input noise magnitude: {}\n'.format(INPUT_NOISE_MAGNITUDE))
    for n in range(1,len(KEEP_PROB)+1):
        fo.write('Dropout keep prob. group {}: {:.2f}\n'.format(n, KEEP_PROB[n]))
    fo.write('Logging frequency: {} global steps\n'.format(LOG_EVERY_N_STEPS))
    fo.write('\nNotes:\n')

# Load network architecture/parameters
G = u.load_graph(SAVE_PATH)
with G.as_default():
    tf.device('/gpu:0')
    saver = tf.train.Saver(var_list=tf.global_variables())
    
    # Get important tensors/operations from graph
    X = G.get_tensor_by_name('input:0')
    Y = G.get_tensor_by_name('output:0')
    labels = G.get_tensor_by_name('labels:0')
    J = G.get_tensor_by_name('loss:0')
    training_op = G.get_operation_by_name('training_op')
    learning_rate = G.get_tensor_by_name('learning_rate:0')
    regularization_parameter = G.get_tensor_by_name('regularization_parameter:0')
    keep_prob = {n:G.get_tensor_by_name('keep_prob_'+str(n)+':0') for n in range(1,len(KEEP_PROB)+1)}
    
    # Load the preprocessed training and validation data in TensorFlow constants if possible so that there is no bottleneck sending things from CPU to GPU
    X_train_ = tf.constant(X_train, dtype=tf.float32)
    Y_train_ = tf.constant(Y_train, dtype=tf.int32)
    del X_train, Y_train
    X_train = X_train_
    Y_train = Y_train_
    
    # Reroute tensors to the location of the data on the GPU, add noise and image augmentation (random crops)
    train_idx = tf.placeholder(tf.int32, shape=[None], name='train_idx')
    input_noise_magnitude = tf.placeholder(tf.float32, shape=[], name='input_noise_magnitude')
    tf.add_to_collection('placeholders', train_idx)
    tf.add_to_collection('placeholders', input_noise_magnitude)
    X_train = tf.gather(X_train, train_idx)
    X_train = tf.pad(X_train, paddings=[[0,0],[3,3],[3,3],[0,0]])
    X_train = u.random_crop(X_train, [32,32,3])
    X_train += input_noise_magnitude*tf.random_normal(tf.shape(X_train), dtype=tf.float32)
    Y_train = tf.gather(Y_train, train_idx)
    ge.reroute_ts([X_train, Y_train], [X, labels])
    
    # Define tensor to compute accuracy metric
    Y_pred = tf.argmax(Y, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(Y_train, Y_pred), tf.float32))
    
    # Start the TF session and load variables
    print('Beginning training...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    with tf.Session(config=config) as sess:
        saver.restore(sess, SAVE_PATH)
        
        # Initialize control flow variables and logs
        max_val_accuracy = -1
        global_steps = 0
        with open(SAVE_PATH+'_val_accuracy.log', 'w+') as fo:
                fo.write('')
        with open(SAVE_PATH+'_val_loss.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_train_accuracy.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_train_loss.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_learning_rate.log', 'w+') as fo:
            fo.write('')
        
        # Iterate over epochs
        for epoch in range(MAX_EPOCHS):
            lr = LEARNING_RATE*np.exp(-(epoch//LEARNING_RATE_ANNEAL_STEP))
            
            # Iterate over batches
            for b in range(m_train//BATCH_SIZE+1):
                
                # Perform forward/backward pass
                slice_lower = b*BATCH_SIZE
                slice_upper = min((b+1)*BATCH_SIZE, m_train)
                feed_dict = {**{learning_rate:lr, train_idx:range(slice_lower, slice_upper), regularization_parameter:REGULARIZATION_PARAMETER, input_noise_magnitude:INPUT_NOISE_MAGNITUDE}, **{keep_prob[n]:KEEP_PROB[n] for n in range(1,len(KEEP_PROB)+1)}}
                train_loss, train_accuracy, _ = sess.run([J, acc, training_op], feed_dict=feed_dict)
                if (train_loss in [np.nan, np.inf]) or (train_loss > 1e3):
                    print('Detected numerical instability in training, exiting')
                    exit()
                
                # Compute metrics, add to logs
                if global_steps % LOG_EVERY_N_STEPS == 0:
                    slice_lower = m_train
                    slice_upper = m_train + VAL_BATCH_SIZE
                    feed_dict = {**{train_idx:range(slice_lower, slice_upper), regularization_parameter:REGULARIZATION_PARAMETER, input_noise_magnitude:INPUT_NOISE_MAGNITUDE}, **{keep_prob[n]:VAL_KEEP_PROB[n] for n in range(1,len(KEEP_PROB)+1)}}
                    val_loss, val_accuracy = sess.run([J, acc], feed_dict=feed_dict)
                    print('Validation loss: {:.2e}, validation accuracy: {:.3f}'.format(val_loss, val_accuracy))
                    with open(SAVE_PATH+'_train_loss.log', 'a') as fo:
                        fo.write(str(train_loss)+'\n')
                    with open(SAVE_PATH+'_train_accuracy.log', 'a') as fo:
                        fo.write(str(train_accuracy)+'\n')
                    with open(SAVE_PATH+'_val_accuracy.log', 'a') as fo:
                        fo.write(str(val_accuracy)+'\n')
                    with open(SAVE_PATH+'_val_loss.log', 'a') as fo:
                        fo.write(str(val_loss)+'\n')
                    with open(SAVE_PATH+'_learning_rate.log', 'a') as fo:
                        fo.write(str(lr)+'\n')
                    
                    # Save if improvement
                    if val_accuracy > max_val_accuracy:
                        min_val_accuracy = val_accuracy
                        print('Saving variables...')
                        saver.save(sess, SAVE_PATH, write_meta_graph=False)
                    
                print('Epoch: {}, batch: {}/{}, loss: {:.2e}, accuracy: {:.3f}, learning_rate: {:.2e}'.format(epoch, b, m_train//BATCH_SIZE, train_loss, train_accuracy, lr))
                
                # Iterate global step
                global_steps += 1
        
        # Print stuff once done
        print('Done!')
























