# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:02:34 2017

Q-learning algorithm to play Pong. Using convolutional autoencoder (CAE) to pretrain the convolutional layers of the network in order to force the filters to learn good representations of the game space. Then, once the convolutional layers have been trained, we then append the rest of of Q-network onto the end of the latent representation layer of the CAE and then do fine-tuning on the higher layers to approximate Q(s,a).

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
import pandas as pd

# Define convolutional autoencoder class
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
    
    Notes: The output length L_out of a convolutional layer is L_out = (L_in-F)/S+1, where F and S are the filter/stride length in that direction. The output length of a transpose convolution (deconvolution) layer is L_out = F + S*(L_in-1). Make sure your ou
    '''
    def __init__(self, input_spec, encoder_spec, decoder_spec, regularization=None):
        self.graph = self.define_graph(input_spec, encoder_spec, decoder_spec, regularization=None)
        self.X = self.graph.get_tensor_by_name('X:0')
        self.Y = self.graph.get_tensor_by_name('Y:0')
        self.Z = self.graph.get_tensor_by_name('Z:0')
        self.Loss = self.graph.get_tensor_by_name('Loss:0')
        self.Lambda = self.graph.get_tensor_by_name('Lambda:0')
    
    # Define the computational graph
    def define_graph(self, input_spec, encoder_spec, decoder_spec, regularization=None):
        
        # Create graph
        G = tf.Graph()
        height, width, depth = input_spec
        with G.as_default():
            
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
                A = tf.nn.relu(tf.nn.conv2d(A, W, strides=stride, padding='VALID') + b, name='A'+str(layer_idx))
                
            # Build latent feature layer (last step of the encoder)
            layer_idx += 1
            filter_dims, stride = encoder_spec[-1]
            h, w, c_in, c_out = filter_dims
            W = tf.Variable(np.random.randn(h, w, c_in, c_out)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
            b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
            Z = tf.add(tf.nn.conv2d(A, W, strides=stride, padding='VALID'), b, name='Z')
            A = tf.nn.relu(Z, name='A'+str(layer_idx))
            
            # Build decoder layers
            for layer in decoder_spec[:-1]:
                layer_idx += 1
                filter_dims, stride, output_dims = layer
                h, w, c_in, c_out = filter_dims
                W = tf.Variable(np.random.randn(h, w, c_out, c_in)*np.sqrt(2/(h*w*c_in+c_out)), dtype=tf.float32, name='W'+str(layer_idx))
                b = tf.Variable(np.zeros((c_out,)), dtype=tf.float32, name='b'+str(layer_idx))
                A = tf.nn.relu(tf.nn.conv2d_transpose(A, W, output_shape=[tf.shape(X)[0], output_dims[0], output_dims[1], output_dims[2]], strides=stride, padding='VALID') + b, name='A'+str(layer_idx))
            
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
                    reg_loss += tf.reduce_sum(G.get_tensor_by_name('W'+str(l))**2)
                reg_loss *= reg_lambda
            elif regularization == 'L1':
                for l in range(1, 1+len(encoder_spec)):
                    reg_loss += tf.reduce_sum(tf.abs(G.get_tensor_by_name('W'+str(l))))
                reg_loss *= reg_lambda
            
            # Calculate loss
            Loss = tf.add(tf.reduce_mean((X-Y)**2), reg_loss, name='Loss')
        
        # Return the computational graph
        return G
    
    # Training function
    def train(self, X_train, lr, max_epochs, batch_size, X_val=None, reload_parameters=False, save_path=None, plot_every_n_steps=25, save_every_n_epochs=2):
        with self.graph.as_default():
            # Define the optimizer and training operation
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(self.Loss)
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
                # Define training metric lists to track
                loss_list = []
                step_list = []
                val_loss_list = []
                plt.ion()
                # Iterate over epochs
                global_step = 0
                nan_flag = False
                for ep in range(max_epochs):
                    # Iterate over minibatches
                    for b, batch in enumerate(minibatches):
                        global_step += 1
                        # Get loss, perform training op
                        _, loss = sess.run([train_op, self.Loss], feed_dict={self.X:batch})
                        print('Episode {}/{}, batch {}/{}, loss: {}'.format(ep+1, max_epochs, b, n_batches, loss))
                        # Exit if nan
                        if loss == np.nan:
                            print('nan error, exiting training')
                            nan_flag = True
                            break
                        # Plot progress
                        if global_step % plot_every_n_steps == 0:
                            loss_list.append(loss)
                            step_list.append(global_step/len(minibatches))
                            plt.figure('Loss')
                            plt.clf()
                            plt.semilogy(step_list, loss_list, label='Training')
                            if X_val is not None:
                                val_loss = sess.run(self.Loss, feed_dict={self.X:X_val})
                                val_loss_list.append(val_loss)
                                plt.semilogy(step_list, val_loss_list, label='Validation')
                                plt.legend()
                            plt.title('Batch loss during training')
                            plt.xlabel('Epoch')
                            plt.ylabel('Avg loss')
                            plt.draw()
                            plt.pause(1e-10)
                    # Save parameters
                    if nan_flag == True:
                        break
                    elif ep % save_every_n_epochs == 0:
                        print('Saving...')
                        saver.save(sess, save_path)
                # Save at end
                print('Saving...')
                saver.save(sess, save_path)
                print('Training complete!')
    
    def visualize_decoded_image(self, X):
        save_str = './checkpoints/'
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
                    y, loss = sess.run([self.Y, self.Loss], feed_dict={self.X:X[m,:,:,:].reshape(1,28,28,1)})
                    # Plot input/output side by side
                    plt.figure('Autoencoder comparison')
                    plt.clf()
                    plt.suptitle('Autoencoder comparison - loss: {}'.format(loss))
                    plt.subplot(121)
                    plt.imshow(X[m,:,:,0])
                    plt.title('Original')
                    plt.subplot(122)
                    plt.imshow(y.reshape(28,28))
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
                        plt.imshow(f)
                plt.draw()

# Define computational graph for Q-network
class QNetwork(object):
    pass

# Gather set of Pong screens to train autoencoder on
def collect_pong_screens():
    pass

# Load dataset
def load_dataset():
    # Load MNIST csv file
    X_raw = pd.read_csv('../../../../Datasets/MNIST/train.csv')
    # Extract training data
    X = (np.array(X_raw.iloc[:,1:])-255/2)/255
    # Reshape training data to [None,28,28,1]
    X = X.reshape((-1,28,28,1))
    # Return array
    return X

def shuffle_dataset(X, val_frac=0.01):
    m = X.shape[0]
    val_idx = int(val_frac*m)
    perm = np.random.permutation(m)
    X_perm = X[perm,:,:,:]
    X_val = X_perm[:val_idx,:,:,:]
    X_train = X_perm[val_idx:,:,:,:]
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

# Train autoencoder on Pong screens
#cae.train()

# Attach Q-network to the end of the autoencoder
#???

# Fine-tune the Q-network
#qnn.train()


''' =========================== TESTING, TESTING ========================= '''

#X = load_dataset()
#X_train, X_val = shuffle_dataset(X, val_frac=0.01)

#cae = ConvolutionalAutoencoder(input_spec=(28,28,1), encoder_spec=[((6,6,1,16),(1,2,2,1)), ((4,4,16,32),(1,4,4,1))], decoder_spec=[((4,4,32,16),(1,2,2,1),(8,8,16)), ((6,6,16,1),(1,3,3,1),(28,28,1))])

cae = ConvolutionalAutoencoder(input_spec=(28,28,1), encoder_spec=[((6,6,1,16),(1,2,2,1)), ((4,4,16,32),(1,4,4,1))], decoder_spec=[((4,4,32,16),(1,2,2,1),(8,8,16)), ((2,2,16,1),(1,3,3,1),(28,28,1))])

cae.train(X_train, lr=1e-3, max_epochs=50, batch_size=500, X_val=X_val, reload_parameters=False, save_path='./checkpoints/MNIST_test', plot_every_n_steps=25, save_every_n_epochs=2)

cae.visualize_decoded_image(X_val)





















