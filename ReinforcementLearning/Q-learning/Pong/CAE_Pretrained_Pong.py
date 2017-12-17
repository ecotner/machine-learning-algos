# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:02:34 2017

Q-learning algorithm to play Pong. Using convolutional autoencoder (CAE) to pretrain the convolutional layers of the network in order to force the filters to learn good representations of the game space. Then, once the convolutional layers have been trained, we then append the rest of of Q-network onto the end of the latent representation layer of the CAE and then do fine-tuning on the higher layers to approximate Q(s,a).

The autoencoder is a several-layer convolutional network that has an encoder and decoder part. The encoder takes pong screens of dimension (160,160,4) in, where the last dimension is the previous 4 iterations of the game. It then adds two channels which are just the (x,y) coordinates of the pixels at each point (normalized to [-1,1]); hopefully this allows the features to propagate the global positions of features to later layers. Each layer of the encoder then downsamples the activation map using max pools and valid padding until it reaches the latent feature layer. The decoder layers are transpose convolutions that upsample the features until the output is the same size as the input. The loss is then calculated by taking the squared difference of the pixel intensities of the input and output.

To do:
    - Add in "advantage function" that subtracts off the mean of Q over many frames, so that the network only has to learn values relative to the mean, and not the absolute value.
    - Make an adaptive gamma so that gamma = exp(-1/n), where n is a running average of the number of frames between rewards.
    x Normalize the pixel coordinate layers to zero mean an unit variance (uniform distribution so sigma^2 = L^2/12, where L is the interval)
    - Figure out optimal CNN layers for autoencoding
        x Add two additional input channels which are just pixel coordinates (x,y); could be that convolution only discerns information about local structure of images, but we also care about the global structure as well.
    x Figure out a way to write plot data to a .txt or .npy file so that I can run the code on dover and then just interrogate the output file. Or maybe even look into using tensorboard.
    x Figure out how to attach encoder layers to Q-network

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
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
        print('Constructing computational graph...')
        self.graph = self.define_graph(input_spec, encoder_spec, decoder_spec, activation, regularization)
        print('Graph done!')
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
                return tf.maximum(x/5.5, x, name=name)
        
        # Create graph
        G = tf.Graph()
        height, width, depth = input_spec
        with G.as_default():
            np.random.seed(seed)
            tf.set_random_seed(seed)
            
            # Input layer
            X = tf.placeholder(dtype=tf.float32, shape=[None,height,width,depth], name='X')
            # Add pixel coordinate layers (will this help??)
            batch_dim = tf.shape(X)[0]
            pix_x, pix_y = np.meshgrid(np.arange(height), np.arange(width))
            pix_x, pix_y = pix_x.reshape(1,height,width,1), pix_y.reshape(1,height,width,1)
            pix_coords = tf.constant((np.concatenate([pix_x, pix_y], axis=-1)-max(height,width)/2)/(max(height,width)*np.sqrt(12)), dtype=tf.float32)
            pix_coords = tf.tile(pix_coords, [batch_dim,1,1,1])
            A = tf.concat([X, pix_coords], axis=-1)
            
            # Build encoder layers
            layer_idx = 0
            for layer in encoder_spec[:-1]:
                layer_idx += 1
                filter_dims, stride, pad_spec = layer
                h, w, c_in, c_out = filter_dims
                W = tf.Variable(np.random.randn(h, w, c_in, c_out)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
                b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
                A = tf.nn.max_pool(a(tf.nn.conv2d(A, W, strides=stride, padding=pad_spec) + b), ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='A'+str(layer_idx))
                
            # Build latent feature layer (last step of the encoder)
            layer_idx += 1
            filter_dims, stride, pad_spec = encoder_spec[-1]
            h, w, c_in, c_out = filter_dims
            W = tf.Variable(np.random.randn(h, w, c_in, c_out)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
            b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
            Z = tf.add(tf.nn.conv2d(A, W, strides=stride, padding=pad_spec), b, name='Z')
            A = a(Z, name='A'+str(layer_idx))
            
            # Build decoder layers
            for layer in decoder_spec[:-1]:
                layer_idx += 1
                filter_dims, stride, pad_spec, output_dims = layer
                h, w, c_in, c_out = filter_dims
                W = tf.Variable(np.random.randn(h, w, c_out, c_in)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
                b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
                A = a(tf.nn.conv2d_transpose(A, W, output_shape=[tf.shape(X)[0], output_dims[0], output_dims[1], output_dims[2]], strides=stride, padding=pad_spec) + b, name='A'+str(layer_idx))
            
            # Build output layer (last layer of decoder)
            layer_idx += 1
            filter_dims, stride, pad_spec, output_dims = decoder_spec[-1]
            h, w, c_in, c_out = filter_dims
            W = tf.Variable(np.random.randn(h, w, c_out, c_in)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
            b = tf.Variable(np.zeros((output_dims[2],)), dtype=tf.float32, name='b'+str(layer_idx))
            Y = tf.add(tf.nn.conv2d_transpose(A, W, output_shape=[tf.shape(X)[0], output_dims[0], output_dims[1], output_dims[2]], strides=stride, padding=pad_spec), b, name='Y')
            
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
    def train(self, X_train, lr0, max_epochs, batch_size, reg_lambda=0, X_val=None, reload_parameters=False, save_path=None, plot_every_n_steps=25, save_every_n_epochs=10000, max_early_stopping_epochs=10):
        '''
        Trains the autoencoder on input images X_train, and plots the loss as it goes along.
        '''
        with self.graph.as_default():
            np.random.seed(seed)
            tf.set_random_seed(seed)
            plt.ioff()
            # Define the optimizer, learning rate decay, gradient clipping, and training operation
            def lr_decay(epoch):
                return lr0/(1+(1+epoch)/(max_epochs/30))
            decay_step = 5
            lr = lr_decay(-1)
            lr_t = tf.placeholder(dtype=tf.float32, shape=[])
            optimizer = tf.train.AdamOptimizer(lr_t)
            gradients, variables = zip(*optimizer.compute_gradients(self.Loss))
            clip_norm = tf.placeholder(dtype=tf.float32, shape=[])
            gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            # Define Saver
            saver = tf.train.Saver()
            # Make minibatches
            print('Making minibatches...')
            minibatches = make_minibatches(X_train, batch_size=batch_size, batch_axis=0)
            print('Minibatches done!')
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
                early_stop_flag = False
                # Iterate over epochs
                for ep in range(max_epochs):
                    # Iterate over minibatches
                    for b, batch in enumerate(minibatches):
                        # Get loss, perform training op
                        _, current_grad, loss = sess.run([train_op, global_norm, self.Loss], feed_dict={self.X:batch, lr_t:lr, self.Lambda:reg_lambda, clip_norm:5*avg_grad})
                        print('Episode {}/{}, batch {}/{}, loss: {}'.format(ep+1, max_epochs, b, n_batches, loss))
                        # Exit if nan
                        if loss == np.nan:
                            print('nan error, exiting training')
                            nan_flag = True
                            break
                        # Update avg gradient norm
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
                                if (val_loss < min_val_loss) and (global_step != 0):
                                    min_val_loss = val_loss
                                    steps_since_min_val_loss = 0
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
                            plt.savefig('Loss.png', bbox_inches='tight')
#                            plt.draw()
#                            plt.pause(1e-10)
                            
                        global_step += 1
                    # Modify learning rate
                    if 1+ep % decay_step == 0:
                        lr = lr_decay(ep)
                    # Save parameters
                    if (nan_flag == True) or (early_stop_flag == True):
                        break
                    elif (ep % save_every_n_epochs == 0) and (ep != 0):
                        print('Saving...')
                        saver.save(sess, save_path)
                # Save at end
#                if (nan_flag == True) or (early_stop_flag == True):
#                    pass
#                else:
#                    print('Saving...')
#                    saver.save(sess, save_path)
#                    print('Training complete!')
    
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
    
    def visualize_conv_filters(self, layer, save_str):
        with self.graph.as_default():
            # Import the weights
            W1 = self.graph.get_tensor_by_name('W'+str(layer)+':0')
            saver = tf.train.Saver(var_list=[W1])
            with tf.Session() as sess:
                saver.restore(sess, save_str)
                # Get the weights of the first convolutional filter layer
                F = sess.run(W1)
                # Figure out how you're going to plot them (height*width subplots arranged in c_in x c_out array?)
                height, width, c_in, c_out = F.shape
                # Plot them all side by side (16 = 4*4)
                plt.figure('Filters')
                plt.clf()
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
    def __init__(self, dense_spec, cae_path, activation='relu', regularization=None):
        ''' Initializes Q-network. '''
        self.graph = self.define_graph(dense_spec, cae_path, activation, regularization)
        self.X = self.graph.get_tensor_by_name('X:0')
        self.YQ = self.graph.get_tensor_by_name('YQ:0')
        self.Z = self.graph.get_tensor_by_name('Z:0')
        self.Q = self.graph.get_tensor_by_name('Q:0')
        self.action = self.graph.get_tensor_by_name('a:0')
        self.QLoss = self.graph.get_tensor_by_name('QLoss:0')
#        self.Lambda = self.graph.get_tensor_by_name('Lambda:0')
    
    def define_graph(self, dense_spec, cae_path, activation='relu', regularization=None):
        
        if activation == 'relu':
            def a(x, name=None):
                return tf.nn.relu(x, name=name)
        elif activation == 'tanh':
            def a(x, name=None):
                return tf.nn.tanh(x, name=name)
        elif activation == 'lrelu':
            def a(x, name=None):
                return tf.maximum(0.1*x, x, name=name)
        
        # Create graph (get autoencoder graph first, then append)
        G = self.load_pretrained_layers(cae_path)
        with G.as_default():
            np.random.seed(seed)
            tf.set_random_seed(seed)
            
            # Get 'input layer' (the latent representation layer of the autoencoder)
            Z = G.get_tensor_by_name('Z:0')
            A = a(tf.contrib.layers.flatten(Z))
            
            # Build encoder layers
            layer_idx = 0
            for layer in dense_spec[:-1]:
                layer_idx += 1
                width, keep_prob = layer
                prev_width = A.get_shape()[-1].value
                W = tf.Variable(tf.random_normal([prev_width, width])*np.sqrt(2/(prev_width + width)), dtype=tf.float32, name='WQ'+str(layer_idx))
                b = tf.Variable(np.zeros((width,)), dtype=tf.float32, name='bQ'+str(layer_idx))
                A = tf.nn.dropout(a(tf.matmul(A, W) + b), keep_prob, name='AQ'+str(layer_idx))
            
            # Build output layer (last dense layer)
            layer_idx += 1
            width = dense_spec[-1]
            prev_width = A.get_shape()[-1].value
            W = tf.Variable(tf.random_normal([prev_width, width])*np.sqrt(2/(prev_width + width)), dtype=tf.float32, name='WQ'+str(layer_idx))
            b = tf.Variable(-1.3*np.ones((width,)), dtype=tf.float32, name='bQ'+str(layer_idx))
            Y = tf.add(tf.matmul(A, W), b, name='YQ')
            
            # Calculate regularization (if applicable)
#            reg_loss = tf.constant(0, dtype=tf.float32)
#            reg_lambda = tf.placeholder(dtype=tf.float32, shape=[], name='Lambda')
#            if regularization == 'L2':
#                for l in range(1, len(dense_spec)):
#                    reg_loss += tf.reduce_sum(G.get_tensor_by_name('WQ'+str(l)+':0')**2)
#                reg_loss *= reg_lambda
#            elif regularization == 'L1':
#                for l in range(1, len(dense_spec)):
#                    reg_loss += tf.reduce_sum(tf.abs(G.get_tensor_by_name('WQ'+str(l)+':0')))
#                reg_loss *= reg_lambda
            
            # Calculate loss
            # NEED TO MAKE Q THE RIGHT SHAPE!
            Q = tf.placeholder(dtype=tf.float32, shape=[None], name='Q')
            action = tf.placeholder(dtype=tf.int32, shape=[None], name='a')
            Q_mask = tf.one_hot(action, depth=3, dtype=tf.float32, axis=-1)
            
            QLoss = tf.reduce_mean((tf.reduce_sum(Q_mask*Y, axis=1)-Q)**2, name='QLoss')
        
        # Return the computational graph
        return G
    
#    def pretrain_conv_layers(self):
#        ''' Trains the first convolutional layers by using an autoencoder to learn appropriate convolutional filters. '''
#        pass
    
    def load_pretrained_layers(self, cae_path):
        ''' Loads the weights for the convolutional layers pretrained on the convolutional autoencoder. Returns the CAE graph and Tensor of the latent representation layer. '''
        # Import metagraph from convolutional autoencoder
        cae_graph = tf.Graph()
        with cae_graph.as_default():
            cae_saver = tf.train.import_meta_graph(cae_path + '.meta')
            # Restore Variables from encoder layers
            with tf.Session() as sess:
                cae_saver.restore(sess, cae_path)
        return cae_graph
    
    def train(self, lr, gamma, max_episodes, batch_size=32, steps_to_skip=1, policy='softmax', epsilon=0.5, reload_parameters=False, save_path=None, plot_every_n_steps=25, save_every_n_episodes=1):
        ''' Trains the Q-network by playing Pong games. '''        
        # Initialize the Pong gym environment, set seeds
        env = gym.make('Pong-v0')
        np.random.seed(seed)
        tf.set_random_seed(seed)
        plt.ioff()
        # Get the computational graph, saver
        with self.graph.as_default():
            saver = tf.train.Saver(var_list=None, max_to_keep=5)                
            # Define optimizer, training op, gradient clipping, etc.
            optimizer = tf.train.GradientDescentOptimizer(lr, name='QGD')
            gradients, variables = zip(*optimizer.compute_gradients(self.QLoss))
            clip_norm = tf.placeholder(dtype=tf.float32, shape=[])
            gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            # Start the Session
            with tf.Session() as sess:
                # Reload/initialize variables
                if reload_parameters:
                    print('Reloading from last checkpoint...')
                    saver.restore(sess, save_path)
                print('Initializing variables...')
                sess.run(tf.global_variables_initializer())
                # Create a frame list and other lists for monitoring stuff
                frame_list = []
                replay_memory = []
                X_val = get_validation_screens()
                Q_val_list = []
                step_list = []
                global_step = 0
                avg_grad = 1e4      # Track avg gradient for clipping
                grad_decay_rate = np.exp(-1/600)
                nan_flag = False    # Keep track of nan values appearing in computation and exit if they occur
                # Iterate over episodes
                global_step = 0
                for ep in range(max_episodes):
                    # Reset the game and get observations, add to frame list, replay memory, etc
                    obs = env.reset()
                    proc_obs = process_raw_frame(obs)
                    add_to_list(frame_list, [np.zeros((1,160,160,1)) for i in range(3)]+[proc_obs], max_to_keep=5)
                    # Iterate over frames
                    done = False
                    frame = 0
                    while done == False:
                        frame += 1
                        # Run observation through Q-network to get prediction
                        y = sess.run(self.YQ, feed_dict={self.X:np.concatenate(frame_list[-4:], axis=-1)})
                        # Decide on action
                        if policy == 'softmax':
                            exp_y = np.exp(y-np.amax(y)).squeeze()**2
                            softmax_y = exp_y/np.sum(exp_y)
                            action = np.random.choice([1,2,3], p=softmax_y)
                        elif policy == 'epsilon-greedy':
                            if np.random.rand() < epsilon*(1 - ep/max_episodes):
                                action = np.random.choice([1,2,3])
                            else:
                                action = np.argmax(y) + 1
                        # Update environment with chosen action, record observation
                        reward_sum = 0
                        for step in range(steps_to_skip+1):
                            if step == 0:
                                obs, reward, done_, _ = env.step(action)
                            else:
                                obs, reward, done_, _ = env.step(1) # Do nothing in between skips, otherwise it overshoots
                            reward_sum += reward
                            if done_ == True:
                                done = True
                        # Add observation, reward, etc to lists
                        add_to_list(frame_list, [process_raw_frame(obs)], max_to_keep=5)
                        # Store data in replay memory
                        self.add_to_replay_memory(frame_list, action, reward_sum, done, replay_memory)
                        # Sample from replay memory, organize into batch
                        s1, s2, a, reward, d = self.sample_from_replay_memory(replay_memory, batch_size)
                        # Calculate Bellman equation update
                        y2 = sess.run(self.YQ, feed_dict={self.X:s2})
                        Q_expected = reward + (1-d)*gamma*np.max(y2, axis=1)
                        # Perform training op on batch
                        _, current_grad, loss = sess.run([train_op, global_norm, self.QLoss], feed_dict={self.X:s1, self.Q:Q_expected, self.action:(a-1), clip_norm:5*avg_grad})
                        # Exit if nan
                        if loss == np.nan:
                            print('nan error, exiting training')
                            nan_flag = True
                            break
                        # Update avg gradient norm
                        avg_grad = (1-grad_decay_rate)*current_grad + grad_decay_rate*avg_grad
                        # Calculate/plot performance metrics
                        mean_batch_loss = np.mean(loss)
                        mean_batch_Q = np.mean(Q_expected)
                        print('Episode: {}/{},\tframe: {},\tbatch loss: {:.3e},\tmean batch Q: {:.3e},\ncurrent max(Q*): {:.3e},\tcurrent action: {},\tcurrent std(Q*)/mean(Q*): {:.3e}'.format(ep+1, max_episodes, frame+1, mean_batch_loss, mean_batch_Q, np.max(y), action, np.std(y)/np.mean(y)))
                        if (global_step % plot_every_n_steps == 0):
                            Q_val = sess.run(self.YQ, feed_dict={self.X:X_val})
                            avg_Q_max = np.mean(np.max(Q_val, axis=-1))
                            Q_val_list.append(avg_Q_max)
                            step_list.append(global_step)
                            plt.figure('Average max Q*')
                            plt.clf()
                            plt.plot(step_list, Q_val_list)
                            plt.xlabel('Global steps')
                            plt.ylabel('<max(Q*)>')
                            plt.title('Average maximum Q* on validation set')
                            plt.savefig('Avg_max_Q.png', bbox_inches='tight')
                        global_step += 1
                    # Save progress after episode ends
                    if nan_flag == True:
                        break
                    elif (ep % save_every_n_episodes == 0) and (ep != 0):
                        print('Saving checkpoint...')
                        saver.save(sess, save_path)
    
    def add_to_replay_memory(self, frame_list, action, reward, done, replay_memory, max_to_keep=5000):
        ''' Adds most recent frames, rewards, and done flags to the replay memory list. '''
        experience = (np.concatenate(frame_list[-5:-1], axis=-1), np.concatenate(frame_list[-4:], axis=-1), action, reward, done)
        add_to_list(replay_memory, [experience])
    
    def sample_from_replay_memory(self, replay_memory, batch_size):
        ''' Samples a batch of experiences from replay memory, then assempbles them into numpy arrays and returns a bunch of stuff. '''
        s1 = []
        s2 = []
        a = []
        r = []
        d = []
        if batch_size >= len(replay_memory):
            perm_idx = np.random.permutation(np.arange(len(replay_memory)))
        else:
            perm_idx = np.random.permutation(np.arange(len(replay_memory)))[:batch_size]
        for idx in perm_idx:
            exp = replay_memory[idx]
            s1.append(exp[0])
            s2.append(exp[1])
            a.append(exp[2])
            r.append(exp[3])
            d.append(exp[4])
        return (np.concatenate(s1, axis=0), np.concatenate(s2, axis=0), np.stack(a), np.stack(r), np.stack(d))
    
    def play(self):
        ''' Plays Pong using learned parameters. '''
        pass

def qnn_play(meta_path, checkpoint_path, mode='softmax'):
    assert (mode in ['softmax', 'max']), 'must choose mode from either max or softmax'
    frames_to_skip = 1
    # Create gym environment
    env = gym.make('Pong-v0')
    # Load the meta graph for Q-network
    G = tf.Graph()
    with G.as_default():
        saver = tf.train.import_meta_graph(meta_path)
        YQ = G.get_tensor_by_name('YQ:0')
        X = G.get_tensor_by_name('X:0')
        with tf.Session() as sess:
            # Load/initialize the parameters from the checkpoint
            saver.restore(sess, checkpoint_path)
            # Iterate over games
            quit_flag = False
            frame_list = []
            while quit_flag == False:
                # Reset environment
                obs = env.reset()
                proc_obs = process_raw_frame(obs)
                add_to_list(frame_list, [np.zeros((1,160,160,1)) for i in range(3)]+[proc_obs], max_to_keep=5)
                # Iterate over frames
                done = False
                frames = 0
                while done == False:
                    frames += 1
                    # Run obervation through Q-network
                    y = sess.run(YQ, feed_dict={X:np.concatenate(frame_list[-4:], axis=-1)})
                    # Select action
                    if mode == 'softmax':
                        exp_y = np.exp(y - np.amax(y)).squeeze()
                        softmax_y = exp_y/np.sum(exp_y)
                        action = np.random.choice([1,2,3], p=softmax_y)
                    elif mode == 'max':
                        action = 1+np.argmax(y)
                    # Take action and update environment
#                    print(action)
                    obs, reward, done, _ = env.step(action)
                    for i in range(frames_to_skip):
                        obs, reward, done, _ = env.step(1)
                    add_to_list(frame_list, [process_raw_frame(obs)], max_to_keep=5)
                    # Render
                    env.render()
                # Ask if we want to play again
                q = input('Press q to quit or anything else to play again: ')
                if q.lower() == 'q':
                    quit_flag = True
            env.render(close=True)
    

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
    print('Saving frames to disk...')
    np.save('./Pong_frames.npy', np.stack(frame_list, axis=-1))
    print('Frames saved!')

def make_validation_screens(val_size=150):
    X_raw = load_pong_dataset()
    perm_idx = np.random.permutation(np.arange(4,X_raw.shape[-1]))[:val_size]
    X_val = np.zeros((val_size,160,160,4))
    print('Creating validation set...')
    for i in range(val_size):
        X_val[i,:,:,:] = X_raw[:,:,perm_idx[i]-4:perm_idx[i]]
    np.save('./val_Pong_frames.npy', X_val)
    print('Done!')

def get_validation_screens():
    X_val = np.load('./val_Pong_frames.npy')
    return X_val

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
    print('Loading pong frames...')
    X_raw = np.load('./Pong_frames.npy')
    print('Frames loaded!')
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
    np.random.seed(seed)
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
    np.random.seed(seed)
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
#collect_pong_screens(max_episodes=5, steps_to_skip=1, max_to_keep=10**10)
# Reload screen data and format for training
#X_train, X_val = shuffle_pong_dataset(load_pong_dataset(), val_frac=0.02)
# Build autoencoder and train on Pong screens
#cae = ConvolutionalAutoencoder(input_spec=(160,160,4), encoder_spec=[((4,4,6,16), (1,1,1,1), 'SAME'), ((4,4,16,16), (1,1,1,1), 'SAME'), ((4,4,16,16), (1,1,1,1), 'SAME')], decoder_spec=[((4,4,16,16), (1,2,2,1), 'SAME', (80,80,16)), ((4,4,16,4), (1,2,2,1), 'SAME', (160,160,4))], activation='lrelu', regularization='L2')
# Encoder layers:
# First layer: (4,4,6,16) filter, (1,1,1,1) stride, same pad, 2x2 max pool, (80,80,16) output
# Second layer: (4,4,6,16) filter, (1,1,1,1) stride, same pad, 2x2 max pool, (40,40,16) output
# Third (latent) layer: (4,4,16,16) filter, (1,1,1,1) stride, same pad, (40,40,16) output
# Decoder layers:
# First layer: (4,4,16,16) filter, (1,2,2,1) stride, same pad, (80,80,16) output
# Second layer: (4,4,16,4) filter, (1,2,2,1) stride, same pad, (160,160,4) output

#cae.train(X_train, lr0=1e-3, max_epochs=50, batch_size=64, reg_lambda=1e-2, X_val=X_val, reload_parameters=False, save_path='./Checkpoints/5/CAE_pretrain', plot_every_n_steps=25, save_every_n_epochs=10**10, max_early_stopping_epochs=10)

#cae.visualize_decoded_image(X_val, save_str='./Checkpoints/5/CAE_pretrain')
#cae.visualize_conv_filters(layer=1, save_str='./Checkpoints/5/CAE_pretrain')

# Attach Q-network to the end of the autoencoder
qnn = QNetwork(dense_spec=[(32, 0.3), (32, 0.3), 3], cae_path='./Checkpoints/5/CAE_pretrain', activation='relu', regularization=None)
# First layer: 32 neurons, 0.3 dropout rate
# Second layer: 32 neurons, 0.3 dropout rate
# Output layer: 3 neurons (corresponding to the 3 available actions)

# Fine-tune the Q-network
qnn.train(lr=1e-7, gamma=np.exp(-1/100), max_episodes=200, batch_size=64, steps_to_skip=1, policy='epsilon-greedy', epsilon=0.5, reload_parameters=True, save_path='./Checkpoints/5/QNN', plot_every_n_steps=50, save_every_n_episodes=1)

# Playtest
#qnn_play(meta_path='./Q_checkpoints_2.meta', checkpoint_path='./Q_checkpoints_2', mode='softmax')










