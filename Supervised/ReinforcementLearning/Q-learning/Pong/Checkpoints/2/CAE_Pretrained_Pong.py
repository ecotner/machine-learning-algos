# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:02:34 2017

Q-learning algorithm to play Pong. Using convolutional autoencoder (CAE) to pretrain the convolutional layers of the network in order to force the filters to learn good representations of the game space. Then, once the convolutional layers have been trained, we then append the rest of of Q-network onto the end of the latent representation layer of the CAE and then do fine-tuning on the higher layers to approximate Q(s,a).

To do:
    1) Add an early stopping detector - if the validation loss does not improve for some specified number of epochs, stop the training. Also save every time the validation error reaches a new minimum.
    2) Figure out how to attach encoder layers to Q-network

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import pandas as pd

seed = 0

''' ===================== DEFINE NEURAL NETWORK CLASSES ==================== '''

# Convolutional autoencoder
class ConvolutionalAutoencoder(object):
    '''
    A class which encapsulates a convolutional autoencoder. Upon initialization, the computational graph is generated, and several attributes are assigned to tensors in the graph for ease.
    
    Inputs:
        input_spec: Tuple (in_height, in_width, in_depth) to specify input layer
        encoder_spec: List of tuples of tuples [(filter_dims, stride), (...), ...], where filter_dims = (height, width, c_in, c_out) and stride = (batch_stride, height_stride, width_stride, depth_stride). The very last element is assigned to the attribute 'Z', and is the output of the latent feature layer.
        decoder_spec: List of tuples of tuples [(filter_dims, stride, output_dims), (...), ...], where filter_dims and stride are defined as in encoder_spec, but output_dims = (out_height, out_width, out_depth). Make sure that c_out and out_depth are the same number, and that the math of the deconvolution works out (see below). The very last element is assigned to the attribute 'Y', and is the end output of the autoencoder.
        regularization: takes either the string 'L1' or 'L2', and sets up the graph to allow for that type of regularization. The regularization parameter is a placeholder Tensor assigned to the attribute 'Lambda'
    
    Attributes:
        X: The input Tensor in the computational graph
        Y: The output Tensor in the computational graph
        Z: The latent feature layer Tensor in the computational graph
        Loss: The loss function Tensor
        Lambda: A placeholder Tensor for specifying regularization strength
    
    Notes: The output length L_out of a convolutional layer is L_out = (L_in-F)/S+1, where F and S are the filter/stride length in that direction. The output length of a transpose convolution (deconvolution) layer is L_out = F + S*(L_in-1). Make sure your output matches your expectation.
    '''
    def __init__(self, input_spec, encoder_spec, decoder_spec, activation='relu', regularization=None):
        self.graph = self.define_graph(input_spec, encoder_spec, decoder_spec, activation, regularization)
        self.X = self.graph.get_tensor_by_name('X:0')
        self.Y = self.graph.get_tensor_by_name('Y:0')
        self.Z = self.graph.get_tensor_by_name('Z:0')
        self.Loss = self.graph.get_tensor_by_name('Loss:0')
        self.Lambda = self.graph.get_tensor_by_name('Lambda:0')
    
    # Define the computational graph
    def define_graph(self, input_spec, encoder_spec, decoder_spec, activation='relu', regularization=None):
        
        if activation == 'relu':
            def a(x, name=None):
                return tf.nn.relu(x, name=name)
        elif activation == 'tanh':
            def a(x, name=None):
                return tf.nn.tanh(x, name=name)
        elif activation == 'lrelu':
            def a(x, name=None):
                return tf.maximum(0.1*x, x, name=name)
        
        # Create graph
        G = tf.Graph()
        height, width, depth = input_spec
        with G.as_default():
            np.random.seed(seed)
            tf.set_random_seed(seed)
            
            # Input layer
            X = tf.placeholder(dtype=tf.float32, shape=[None,height,width,depth], name='X')
            A = X
            
            # Build encoder layers
            layer_idx = 0
            for layer in encoder_spec[:-1]:
                layer_idx += 1
                filter_dims, stride = layer
                h, w, c_in, c_out = filter_dims
                W = tf.Variable(np.random.randn(h, w, c_in, c_out)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
                b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
                A = tf.nn.max_pool(a(tf.nn.conv2d(A, W, strides=stride, padding='VALID') + b), ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='A'+str(layer_idx))
                
            # Build latent feature layer (last step of the encoder)
            layer_idx += 1
            filter_dims, stride = encoder_spec[-1]
            h, w, c_in, c_out = filter_dims
            W = tf.Variable(np.random.randn(h, w, c_in, c_out)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
            b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
            Z = tf.add(tf.nn.conv2d(A, W, strides=stride, padding='VALID'), b, name='Z')
            A = a(Z, name='A'+str(layer_idx))
            
            # Build decoder layers
            for layer in decoder_spec[:-1]:
                layer_idx += 1
                filter_dims, stride, output_dims = layer
                h, w, c_in, c_out = filter_dims
                W = tf.Variable(np.random.randn(h, w, c_out, c_in)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
                b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
                A = a(tf.nn.conv2d_transpose(A, W, output_shape=[tf.shape(X)[0], output_dims[0], output_dims[1], output_dims[2]], strides=stride, padding='VALID') + b, name='A'+str(layer_idx))
            
            # Build output layer (last layer of decoder)
            layer_idx += 1
            filter_dims, stride, output_dims = decoder_spec[-1]
            h, w, c_in, c_out = filter_dims
            W = tf.Variable(np.random.randn(h, w, c_out, c_in)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
            b = tf.Variable(np.zeros((output_dims[2],)), dtype=tf.float32, name='b'+str(layer_idx))
            Y = tf.add(tf.nn.conv2d_transpose(A, W, output_shape=[tf.shape(X)[0], output_dims[0], output_dims[1], output_dims[2]], strides=stride, padding='VALID'), b, name='Y')
            
            # Calculate regularization (if applicable)
            reg_loss = tf.constant(0, dtype=tf.float32)
            reg_lambda = tf.placeholder(dtype=tf.float32, shape=[], name='Lambda')
            if regularization == 'L2':
                for l in range(1, 1+len(encoder_spec)):
                    reg_loss += tf.reduce_sum(G.get_tensor_by_name('W'+str(l)+':0')**2)
                reg_loss *= reg_lambda
            elif regularization == 'L1':
                for l in range(1, 1+len(encoder_spec)):
                    reg_loss += tf.reduce_sum(tf.abs(G.get_tensor_by_name('W'+str(l)+':0')))
                reg_loss *= reg_lambda
            
            # Calculate loss
            Loss = tf.add(tf.reduce_mean((X-Y)**2), reg_loss, name='Loss')
        
        # Return the computational graph
        return G
    
    # Training function
    def train(self, X_train, lr, max_epochs, batch_size, reg_lambda=0, X_val=None, reload_parameters=False, save_path=None, plot_every_n_steps=25, save_every_n_epochs=10000):
        '''
        Trains the autoencoder on input images X_train, and plots the loss as it goes along.
        '''
        with self.graph.as_default():
            np.random.seed(seed)
            tf.set_random_seed(seed)
            plt.ion()
            # Define the optimizer and training operation
            optimizer = tf.train.AdamOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.Loss))
            clip_norm = tf.placeholder(dtype=tf.float32, shape=[])
            gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
#            train_op = optimizer.minimize(self.Loss)
            # Define Saver
            saver = tf.train.Saver()
            # Make minibatches
            minibatches = make_minibatches(X_train, batch_size=batch_size, batch_axis=0)
            n_batches = len(minibatches)
            # Create TF Session
            with tf.Session() as sess:
                # Initialize variables or load parameters
                if reload_parameters == True:
                    saver.restore(sess, save_path)
                else:
                    sess.run(tf.global_variables_initializer())
                # Define training metrics and quantities to track
                loss_list = []
                step_list = []
                val_loss_list = []
                global_step = 0
                avg_grad = 1e4      # Track avg gradient for clipping
                grad_decay_rate = np.exp(-1/len(minibatches))
                nan_flag = False    # Keep track of nan values appearing in computation and exit if they occur
                min_val_loss = 1e10     # Keep track of validation loss for early stopping
                steps_since_min_val_loss = 0
                max_early_stopping_epochs = 2
                early_stop_flag = False
                # Iterate over epochs
                for ep in range(max_epochs):
                    # Iterate over minibatches
                    for b, batch in enumerate(minibatches):
                        # Get loss, perform training op
                        _, current_grad, loss = sess.run([train_op, global_norm, self.Loss], feed_dict={self.X:batch, self.Lambda:reg_lambda, clip_norm:5*avg_grad})
                        print('Episode {}/{}, batch {}/{}, loss: {}'.format(ep+1, max_epochs, b, n_batches, loss))
                        # Exit if nan
                        if loss == np.nan:
                            print('nan error, exiting training')
                            nan_flag = True
                            break
                        avg_grad = (1-grad_decay_rate)*current_grad + grad_decay_rate*avg_grad
                        # Plot progress
                        if global_step % plot_every_n_steps == 0:
                            loss_list.append(loss)
                            step_list.append(global_step/len(minibatches))
                            plt.figure('Loss')
                            plt.clf()
                            plt.semilogy(step_list, loss_list, label='Training')
                            if X_val is not None:
                                val_loss = sess.run(self.Loss, feed_dict={self.X:X_val, self.Lambda:reg_lambda})
                                val_loss_list.append(val_loss)
                                plt.semilogy(step_list, val_loss_list, label='Validation')
                                plt.legend()
                                if val_loss < min_val_loss:
                                    print('Saving optimal solution...')
                                    saver.save(sess, save_path)
                                elif steps_since_min_val_loss < max_early_stopping_epochs*len(minibatches)/plot_every_n_steps:
                                    steps_since_min_val_loss += 1
                                else:
                                    early_stop_flag = True
                                    print('No progress on validation loss in last {} epochs, exiting training.'.format(max_early_stopping_epochs))
                                    break
                            plt.title('Batch loss during training')
                            plt.xlabel('Epoch')
                            plt.ylabel('Avg loss')
                            plt.draw()
                            plt.pause(1e-10)
                        global_step += 1
                    # Save parameters
                    if (nan_flag == True) or (early_stop_flag == True):
                        break
                    elif (ep % save_every_n_epochs == 0) and (ep != 0):
                        print('Saving...')
                        saver.save(sess, save_path)
                # Save at end
                print('Saving...')
                saver.save(sess, save_path)
                print('Training complete!')
    
    def visualize_decoded_image(self, X, save_str='./checkpoints/'):
        ''' Compares the autoencoded image with the original side-by-side. '''
        with self.graph.as_default():
            # Create Saver
            saver = tf.train.Saver()
            # Start TF Session
            with tf.Session() as sess:
                # Restore parameters
                saver.restore(sess, save_str)
                # Iterate over input data
                for m in range(X.shape[0]):
                    # Evaluate autoencoder output
                    y, loss = sess.run([self.Y, self.Loss], feed_dict={self.X:X[m,:,:,:].reshape(1,160,160,4), self.Lambda:0})
                    # Plot input/output side by side
                    plt.figure('Autoencoder comparison')
                    plt.clf()
                    plt.suptitle('Autoencoder comparison - loss: {}'.format(loss))
                    plt.subplot(121)
                    plt.imshow(X[m,:,:,0], cmap='gray')
                    plt.title('Original')
                    plt.subplot(122)
                    plt.imshow(1+y[0,:,:,0].reshape(160,160), cmap='gray')
                    plt.title('Reconstructed')
                    plt.draw()
                    plt.pause(1e-10)
                    q = input('Type q to quit or any other button to continue: ')
                    if q.lower() == 'q':
                        break
    
    def visualize_conv_filters(self, save_str):
        with self.graph.as_default():
            # Import the weights
            W1 = self.graph.get_tensor_by_name('W1:0')
            saver = tf.train.Saver(var_list=[W1])
            with tf.Session() as sess:
                saver.restore(sess, save_str)
                # Get the weights of the first convolutional filter layer
                F = sess.run(W1)
                # Figure out how you're going to plot them (height*width subplots arranged in c_in x c_out array?)
                height, width, c_in, c_out = F.shape
                # Plot them all side by side (16 = 4*4)
                plt.figure('Filters')
                for i in range(c_in):
                    for j in range(c_out):
                        f = F[:,:,i,j]
                        plt.subplot(c_out, c_in, 1+c_out*i+j)
                        plt.imshow(f, cmap='gray')
                plt.draw()

# Define computational graph for Q-network
class QNetwork(object):
    '''
    A class which encapsulates a pong-playing Q-network. The network consists of a CNN which takes (preprocessed) images of the playing screen as input, and outputs a vector of values which give the expected discounted reward of the next state for each possible action. In this sense, the network acts as a function approximator for the value function Q(s,a).
    '''
    def __init__(self, input_spec, encoder_spec, decoder_spec, activation='relu', regularization=None):
        ''' Initializes Q-network. '''
        self.graph = self.define_graph(input_spec, encoder_spec, decoder_spec, activation, regularization)
        self.X = self.graph.get_tensor_by_name('X:0')
        self.Y = self.graph.get_tensor_by_name('Y:0')
        self.Z = self.graph.get_tensor_by_name('Z:0')
        self.Loss = self.graph.get_tensor_by_name('Loss:0')
        self.Lambda = self.graph.get_tensor_by_name('Lambda:0')
    
    def define_graph(self, input_spec, conv_spec, dense_spec, activation='relu', regularization=None):
        
        if activation == 'relu':
            def a(x, name=None):
                return tf.nn.relu(x, name=name)
        elif activation == 'tanh':
            def a(x, name=None):
                return tf.nn.tanh(x, name=name)
        elif activation == 'lrelu':
            def a(x, name=None):
                return tf.maximum(0.1*x, x, name=name)
        
        # Create graph
        G = tf.Graph()
        height, width, depth = input_spec
        with G.as_default():
            np.random.seed(seed)
            tf.set_random_seed(seed)
            
            # Input layer
            X = tf.placeholder(dtype=tf.float32, shape=[None,height,width,depth], name='X')
            A = X
            
            # Build encoder layers
            layer_idx = 0
            for layer in conv_spec[:-1]:
                layer_idx += 1
                filter_dims, stride = layer
                h, w, c_in, c_out = filter_dims
                W = tf.Variable(np.random.randn(h, w, c_in, c_out)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
                b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
                A = tf.nn.max_pool(a(tf.nn.conv2d(A, W, strides=stride, padding='VALID') + b), ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='A'+str(layer_idx))
                
            # Build latent feature layer (last step of the encoder)
            layer_idx += 1
            filter_dims, stride = conv_spec[-1]
            h, w, c_in, c_out = filter_dims
            W = tf.Variable(np.random.randn(h, w, c_in, c_out)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
            b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
            Z = tf.add(tf.nn.conv2d(A, W, strides=stride, padding='VALID'), b, name='Z')
            A = a(Z, name='A'+str(layer_idx))
            A = tf.contrib.layers.flatten(A)
            
            # Build dense layers
            for c_out in dense_spec[:-1]:
                layer_idx += 1
                c_in = tf.shape(A)[-1]
                W = tf.Variable(tf.random_normal((c_in, c_out))*np.sqrt(2/(c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
                b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
                A = a(tf.matmul(A, W) + b, name='A'+str(layer_idx))
            
            # Build output layer (last dense layer)
            layer_idx += 1
            c_out = decoder_spec[-1]
            c_in = tf.shape(A)[-1]
            W = tf.Variable(tf.random_normal((c_in, c_out))*np.sqrt(2/(c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
            b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
            Y = tf.add(tf.matmul(A, W), b, name='Y')
            
            # Calculate regularization (if applicable)
#            reg_loss = tf.constant(0, dtype=tf.float32)
#            reg_lambda = tf.placeholder(dtype=tf.float32, shape=[], name='Lambda')
#            if regularization == 'L2':
#                for l in range(1, 1+len(conv_spec)):
#                    reg_loss += tf.reduce_sum(G.get_tensor_by_name('W'+str(l)+':0')**2)
#                reg_loss *= reg_lambda
#            elif regularization == 'L1':
#                for l in range(1, 1+len(conv_spec)):
#                    reg_loss += tf.reduce_sum(tf.abs(G.get_tensor_by_name('W'+str(l)+':0')))
#                reg_loss *= reg_lambda
            
            # Calculate loss
            # NEED TO MAKE Q THE RIGHT SHAPE!
            Q = tf.placeholder(dtype=tf.float32, shape=[None], name='Q')
            a = tf.placeholder(dtype=tf.int32, shape=[None], name='a')
            Q_mask = tf.one_hot(a, depth=3, dtype=tf.int32, axis=-1)
            
            Loss = tf.add(tf.reduce_mean((tf.reduce_sum(Q_mask*Y, axis=1)-Q)**2), reg_loss, name='Loss')
        
        # Return the computational graph
        return G
    
    def pretrain_conv_layers(self):
        ''' Trains the first convolutional layers by using an autoencoder to learn appropriate convolutional filters. '''
        pass
    
    def load_pretrained_layers(self):
        ''' Loads the weights for the convolutional layers pretrained on the convolutional autoencoder. '''
        pass
    
    def train(self):
        ''' Trains the Q-network by playing Pong games. '''
        pass
    
    def play(self):
        ''' Plays Pong using learned parameters. '''
        pass
    
    

''' ==================== DATA PROCESSING AND HANDLING ==================== '''

def process_raw_frame(input_data, color=False, crop=True, ds_factor=1):
    ''' Takes the raw image data, crops it down so that only the play space (160x160 box) is visible, takes a single slice of the RGB color channels (channel 2 seems to have the most contrast), and then downsamples it.
    Args:
        input_data: The raw data, in a 210x160x3 array.
        ds_factor: the factor by which to downsample. Must be integer i >= 1 which is a divisor of 160 (1,2,4,5,8,10,16,20,32,40,80,160).
        crop: Whether or not to crop to 160x160 array
    Output:
        processed_data: An array of the processed output data with shape (1,height,width,color).
        '''
    if color == False:
        processed_data = input_data[:,:,1]
        if crop == True:
            processed_data = processed_data[34:194,:]
        m, n = processed_data.shape
        if ds_factor != 1:
            processed_data_temp = np.zeros((m//ds_factor,n//ds_factor))
            # Is there some way to vectorize this? (not necessary if ds_factor=1 though)
            for i in range(m//ds_factor):
                for j in range(n//ds_factor):
                    processed_data_temp[i,j] = np.max(processed_data[ds_factor*i:ds_factor*(i+1),ds_factor*j:ds_factor*(j+1)])
            processed_data = processed_data_temp
        out_shape = (1,m,n,1)
    else:
        processed_data = input_data[:,:,:]
        if crop == True:
            processed_data = processed_data[34:194,:,:]
        m, n, _ = processed_data.shape
        if ds_factor != 1:
            processed_data_temp = np.zeros((m//ds_factor,n//ds_factor,3))
            for i in range(m//ds_factor):
                for j in range(n//ds_factor):
                    for k in range(3):
                        processed_data_temp[i,j,k] = np.max(processed_data[ds_factor*i:ds_factor*(i+1),ds_factor*j:ds_factor*(j+1),k])
            processed_data = processed_data_temp
        out_shape = (1,m,n,3)
    return (processed_data.reshape(out_shape)-115)/230 # Normalizes pixel values

def add_to_list(existing_list, elements_to_add, max_to_keep=10**4):
    ''' Adds a set of elements to the list existing_list. Modifies the existing_list IN PLACE, so doesn't return anything. Assumes elements_to_add is a LIST.'''
    assert type(elements_to_add) == list, 'elements_to_add must be a list'
    if len(existing_list) + len(elements_to_add) <= max_to_keep:
        for frame in elements_to_add:
            existing_list.append(frame)
    else:
        for frame in elements_to_add:
            existing_list.append(frame)
        diff = len(existing_list) - max_to_keep
        for i in range(diff):
            existing_list.pop(0)

def add_to_array(array, array_to_add, max_to_keep=10**4, axis=-1):
    ''' Appends array_to_add to array in place. '''
    assert type(array) == np.ndarray, 'array_to_add must be np.ndarray'
    if type(array_to_add) != np.ndarray:
        temp_array = np.append(array, np.array([array_to_add]))
    else:
        temp_array = np.append(array, array_to_add, axis=axis)
    diff = temp_array.shape[axis] - max_to_keep
    if diff > 0:
        temp_array = np.delete(temp_array, np.s_[:diff], axis=axis)
    array[:] = temp_array


# Gather set of Pong screens to train autoencoder on
def collect_pong_screens(max_episodes, steps_to_skip=1, max_to_keep=10**4):
    # Initialize gym environment
    env = gym.make('Pong-v0')
    # Iterate over episodes
    frame_list = []
#    global_step = 0
    for ep in range(max_episodes):
        # Add initial frames
        add_to_list(frame_list, [np.zeros((160,160)) for i in range(3)], max_to_keep=max_to_keep)
        obs = env.reset()
        proc_obs = process_raw_frame(obs).squeeze()
        add_to_list(frame_list, [proc_obs], max_to_keep=max_to_keep)
        # Iterate over frames
        done = False
        while done == False:
            # Select random action
            action = np.random.choice([1,2,3])
            # Make action
            for i in range(steps_to_skip):
                if i == 0:
                    obs, reward, done, _ = env.step(action)
                else:
                    # Do nothing in between action steps
                    obs, reward, done, _ = env.step(1)
            # Store processed observation in a list of frames
            proc_obs = process_raw_frame(obs).squeeze()
            add_to_list(frame_list, [proc_obs], max_to_keep=max_to_keep)
#            global_step += 1
            print('Number of frames collected: {}'.format(len(frame_list)))
    # Save frames to disk
    np.save('./Pong_frames.npy', np.stack(frame_list, axis=-1))

# Load MNIST dataset
def load_MNIST_dataset():
    # Load MNIST csv file
    X_raw = pd.read_csv('../../../../Datasets/MNIST/train.csv')
    # Extract training data
    X = (np.array(X_raw.iloc[:,1:])-255/2)/255
    # Reshape training data to [None,28,28,1]
    X = X.reshape((-1,28,28,1))
    # Return array
    return X

# Load the collected Pong screens
def load_pong_dataset():
    X_raw = np.load('../../Pong_frames.npy')
    return X_raw

def shuffle_MNIST_dataset(X, val_frac=0.01):
    m = X.shape[0]
    val_idx = int(val_frac*m)
    perm = np.random.permutation(m)
    X_perm = X[perm,:,:,:]
    X_val = X_perm[:val_idx,:,:,:]
    X_train = X_perm[val_idx:,:,:,:]
    return X_train, X_val

def shuffle_pong_dataset(X, val_frac=0.01):
    '''
    Takes in an array of shape (160,160,m_train+m_val+3) containing pong screens, temporially-indexed in the 2 axis. Returns training and validation arrays of dimension (m_,160,160,4) containing all possible 4-frame sequences.
    '''
    m = X.shape[-1]-3
    val_idx = int(val_frac*m)+1
    perm = np.random.permutation(m)
    perm_train = perm[val_idx:]
    perm_val = perm[:val_idx]
    X_train = np.zeros((m-val_idx,160,160,4))
    X_val = np.zeros((val_idx,160,160,4))
    for i, j in zip(range(val_idx), perm_val):
        X_val[i,:,:,:] = X[:,:,j:j+4]
    for i, j in zip(range(m-val_idx), perm_train):
        X_train[i,:,:,:] = X[:,:,j:j+4]
    return X_train, X_val

# Make minibatches of training data
def make_minibatches(X, batch_size, batch_axis=0):
    batches = []
    m = X.shape[batch_axis]
    perm_idx = np.random.permutation(m)
    X_perm = X[perm_idx,:,:,:]
    for b in range(m//batch_size):
        minibatch = X_perm[b*batch_size:(b+1)*batch_size,:,:,:]
        batches.append(minibatch)
    if m % batch_size != 0:
        minibatch = X_perm[(b+1)*batch_size:,:,:,:]
        batches.append(minibatch)
    return batches





''' =========================== TESTING, TESTING ========================= '''

''' # Test autoencoder on MNIST dataset
X = load_MNIST_dataset()
X_train, X_val = shuffle_MNIST_dataset(X, val_frac=0.01)

cae = ConvolutionalAutoencoder(input_spec=(28,28,1), encoder_spec=[((6,6,1,16),(1,2,2,1)), ((4,4,16,32),(1,4,4,1))], decoder_spec=[((4,4,32,16),(1,2,2,1),(8,8,16)), ((6,6,16,1),(1,3,3,1),(28,28,1))])

cae = ConvolutionalAutoencoder(input_spec=(28,28,1), encoder_spec=[((6,6,1,16),(1,2,2,1)), ((4,4,16,32),(1,4,4,1))], decoder_spec=[((4,4,32,16),(1,2,2,1),(8,8,16)), ((2,2,16,1),(1,3,3,1),(28,28,1))])

cae.train(X_train, lr=1e-3, max_epochs=50, batch_size=500, X_val=X_val, reload_parameters=False, save_path='./checkpoints/MNIST_test', plot_every_n_steps=25, save_every_n_epochs=2)

cae.visualize_decoded_image(X_val)
'''


# Pre-train autoencoder on Pong screens
# Collect pong screen data
#collect_pong_screens(max_episodes=3, steps_to_skip=1, max_to_keep=10**4)
# Reload screen data and format for training
#X_train, X_val = shuffle_pong_dataset(load_pong_dataset(), val_frac=0.03)
# Build autoencoder and train on Pong screens
cae = ConvolutionalAutoencoder(input_spec=(160,160,4), encoder_spec=[((5,5,4,16), (1,1,1,1)), ((5,5,16,32), (1,1,1,1)), ((3,3,32,64), (1,1,1,1))], decoder_spec=[((3,3,64,32), (1,1,1,1), (37,37,32)), ((6,6,32,16), (1,2,2,1), (78,78,16)), ((6,6,16,4), (1,2,2,1), (160,160,4))], activation='lrelu', regularization='L2')
# Encoder layers:
# First layer: (5,5,4,16) filter, (1,1,1,1) stride, 2x2 max pool, (78,78,16) output
# Second layer: (5,5,16,32) filter, (1,1,1,1) stride, 2x2 max pool, (37,37,32) output
# Third layer: (3,3,32,64) filter, (1,1,1,1) stride, (35,35,64) output
# Decoder layers:
# First layer: (3,3,64,32) filter, (1,1,1,1) stride, (37,37,32) output
# Second layer: (6,6,32,16) filter, (1,2,2,1) stride, (78,78,16) output
# Third layer: (6,6,16,4) filter, (1,2,2,1) stride, (160,160,4) output

cae.train(X_train, lr=1e-3, max_epochs=200, batch_size=32, reg_lambda=1e-2, X_val=X_val, reload_parameters=False, save_path='./checkpoints', plot_every_n_steps=25, save_every_n_epochs=2)

#cae.visualize_decoded_image(X_val, save_str='./checkpoints')
#cae.visualize_conv_filters(save_str='./checkpoints')

# Attach Q-network to the end of the autoencoder
#???

# Fine-tune the Q-network
#qnn.train()














