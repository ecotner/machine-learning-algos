# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 19:53:13 2018

Constructs a ResNet computational graph

@author: Eric Cotner
"""

#import numpy as np
import tensorflow as tf

class ResNet(object):
    '''
    Class containing methods to construct the computational graph for a residual network (ResNet), containing skip connections and Inception blocks. Has a similar functionality to Keras, but just uses TensorFlow.
    
    By default, tensors of all layers can be accessed from the graph (ResNet.G) using the Graph.get_tensor_by_name() method. The default naming scheme is:
        '<weight type>_<inception>_<layer_type>_<layer index>:0'
    where all the layers within an inception block have the same layer index. So for example, the tensor representing the filter for a 3x3 convolutional layer within an inception block at layer 2 would have the name 'W_inception_conv_3x3_2:0' (biases start with 'b' rather than 'W'). The tensor representing the output of a 2x2 max pool located at the third layer would have the name 'max_pool_2x2_3:0'. The output of the concatenated filters from the inception blocks themselves have the form:
        inception_block_<layer index>:0
    so that the output of the inception block referenced above would simply be 'inception_block_2:0'.
    The class can also build skip connections by using the ResNet.start_skip_connection() and ResNet.end_skip_connection() methods. Make sure that the dimensions of the neurons are the same at both the start and end segments of the skip connection.
    
    Arguments:
        dtype: dtype of the input tensor (use TensorFlow types)
        shape: shape of the input tensor
        name: an optional name for the input (default is 'X')
    
    Attributes:
        G: the computational graph
        input: the input tensor
        output: the (most recent) output tensor
        labels: the target labels for classification
        skip_start = keeps track of the node at which the skip connection starts
        layer_num: the layer index of the (most_recent) output tensor (layer indexing starts at 1, and skip connections and flattening operations don't count towards it)
    
    Methods:
        get_graph: returns the computational graph ResNet.G
        get_tensor: returns tensor from the computational graph by name
        reset_graph: resets the graph to a blank state
        activation: applies an activation function
        add_conv_layer: adds a convolutional layer to the most recent output
        add_dense_layer: adds a dense/fully-connected layer to the most recent output
        add_max_pool_layer: adds a max pooling layer to the most recent output
        flatten_layer flattens the most recent output to a form appropriate for attaching dense layers
        add_inception_block: adds an inception block composed of multiple convolutional filters
        start_skip_connection: marks the root node of a skip connection
        end_skip_connection: adds the most recent output layer to the layer marked by the start_skip_connection method
        define_training_op: defines the optimization method and training operation
    '''
    
    def __init__(self, dtype, shape, name='input'):
        ''' Initialize the graph with a single input layer. '''
        self.G = tf.Graph()
        with self.G.as_default():
            self.input = tf.placeholder(dtype, shape, name)
            self.output = self.input
            self.labels = None
            self.loss = None
            self.learning_rate = None
            self.training_op = None
            self.skip_start = None
            self.layer_num = 1
    
    def get_graph(self):
        ''' Returns the computational graph. '''
        return self.G
    
    def get_tensor(self, name):
        ''' Returns tensor from the computational graph by name. '''
        return self.G.get_tensor_by_name(name+':0')
    
    def reset_graph(self, dtype, shape, name='X'):
        ''' Resets the computational graph by calling __init__() method. '''
        self.__init__(dtype, shape=None, name=name)
    
    def activation(self, A, activation, name):
        ''' Applies the activation function <activation> to the tensor A. '''
        with self.G.as_default():
            if activation.lower() == 'relu':
                return tf.nn.relu(A, name=name)
            elif activation.lower() == 'softmax':
                return tf.nn.softmax(A, name=name)
            elif activation.lower() == 'none':
                return tf.identity(A, name=name)
            elif activation.lower() == 'sigmoid':
                return tf.nn.sigmoid(A, name=name)
            elif activation.lower() == 'tanh':
                return tf.nn.tanh(A, name=name)
            else:
                raise Exception('unknown activation function')
    
    def add_conv_layer(self, f, n_c, s, padding='SAME', activation='relu', use_cudnn_on_gpu=None, data_format=None, name='conv'):
        '''
        Adds a convolutional layer to the network. All filter weights have Xavier initialization, and biases are initialized to zeros.
        Arguments:
            f: the dimension of the filter (expects square input)
            n_c: the number of output channels (or number of filters to be applied)
            s: the stride
            padding: whether to use 'SAME' or 'VALID' padding
            activation: the type of activation
            use_cudnn_on_gpu: whether to use cuDNN on your GPU
            data_format: specifies the data format of the input/output channels. Default is 'NHWC', with alternative option 'NCHW'
            name: an optional name for the convolutional layer (the layer index will be appended to this)
        '''
        with self.G.as_default():
            n_in = self.output.shape.as_list()[-1]
            W = tf.Variable(initial_value=tf.random_normal(shape=[f,f,n_in,n_c], dtype=tf.float32)*tf.sqrt(2/(f*f*n_in + n_c)), dtype=tf.float32, name='W_'+name+'_{0}x{0}_{1}'.format(f, self.layer_num))
            b = tf.Variable(initial_value=tf.zeros(shape=[n_c]), dtype=tf.float32, name='b_'+name+'_{0}x{0}_{1}'.format(f, self.layer_num))
            A = tf.nn.conv2d(self.output, W, [1,s,s,1], padding, use_cudnn_on_gpu=None, data_format=None) + b
            name = name+'_{0}x{0}_{1}'.format(f, self.layer_num)
            self.output = self.activation(A, activation, name)
            self.layer_num += 1
    
    def add_dense_layer(self, width, activation='relu', name='dense'):
        '''
        Adds a dense/fully-connected layer to the network. Weights have Xavier initialization and biases are initialized to zeros.
        Arguments:
            width: the number of neurons in the network
            activation: the type of activation
            name: an optional name for the layer (the layer index will be appended to this)
        '''
        with self.G.as_default():
            n_in = self.output.shape.as_list()[-1]
            W = tf.Variable(initial_value=tf.random_normal(shape=[n_in,width], dtype=tf.float32)*tf.sqrt(2/(n_in + width)), name='W_'+name+str(self.layer_num))
            b = tf.Variable(initial_value=tf.zeros(shape=[width]), dtype=tf.float32, name='b_'+name+str(self.layer_num))
            A = tf.matmul(self.output, W) + b
            name = name+str(self.layer_num)
            self.output = self.activation(A, activation, name)
            self.layer_num += 1
    
    def add_output_layer(self, width, activation='relu', name='output'):
        '''
        Adds a dense/fully-connected layer to the network. Weights have Xavier initialization and biases are initialized to zeros.
        Arguments:
            width: the number of neurons in the network
            activation: the type of activation
            name: an optional name for the layer (the layer index will be appended to this)
        '''
        with self.G.as_default():
            n_in = self.output.shape.as_list()[-1]
            W = tf.Variable(initial_value=tf.random_normal(shape=[n_in,width], dtype=tf.float32)*tf.sqrt(2/(n_in + width)), name='W_'+name+str(self.layer_num))
            b = tf.Variable(initial_value=tf.zeros(shape=[width]), dtype=tf.float32, name='b_'+name+str(self.layer_num))
            A = tf.matmul(self.output, W) + b
            self.output = self.activation(A, activation, name)
            self.layer_num += 1

    def add_max_pool_layer(self, f, s, activation='none', name='max_pool'):
        '''
        Adds a max pool layer to the network.
        Arguments:
            f: dimensions of square filter
            s: stride
            activation: the type of activation function
            name: an optional name (the layer index will be appended to this)
        '''
        with self.G.as_default():
            A = tf.nn.max_pool(self.output, [1,f,f,1], [1,s,s,1], padding='SAME')
            name = name+'_{0}x{0}_{1}'.format(f, self.layer_num)
            self.output = self.activation(A, activation, name)
            self.layer_num += 1
    
    def flatten_layer(self):
        ''' Flattens the output of a previous covolutional layer so that fully connected layers can be attached. '''
        with self.G.as_default():
            self.output = tf.contrib.layers.flatten(self.output)

    def add_inception_block(self, conv_filter_sizes, num_conv_filters, stride=1, max_pool_filter_sizes=[], num_max_pool_filters=[], activation='relu', shield_channels=True, name='inception'):
        '''
        Creates an inception block composed of multiple convolutional filters and max pool layers, concatenated into a single tensor. In order to reduce computation, each convolutional filter larger than 1x1 can be preceded by a 1x1 convolutional filter with output channels equal to <shield_channels>. This can save on computation if the number of shield channels is less than (2f^2*l^2*n_in*n_out+1)/(2l^2(n_in+f^2*n_out)).
        Arguments:
            conv_filter_sizes: a list of dimensions of square convolutional filters
            num_conv_fiters: the number of filters of each type to be applied
            stride: the (shared) stride of each filter (all strides must be the same so the concatenation operation is well-defined)
            max_pool_filter_sizes: list of dimensions of square max pool filters
            num_max_pool_filters: the number of filters coming from the max pool channel (a 1x1 convolution is applied after max pool to change the channel dimension)
            activation: the type of activation function
            shield_channels: number of 1x1 convolutions to do before larger fxf convolutions to reduce the number of input channels and save computation. Options are True (automatically chooses a number of filters), False (doesn't perform any 1x1 convolutions), and a arbitrary positive integer number
            name: an optional name for the inception block
        '''
        assert all([(f>0) for f in conv_filter_sizes]), 'some conv_filter_sizes are negative'
        assert (shield_channels == True) or (shield_channels == False) or (shield_channels > 0)
        with self.G.as_default():
            outputs = []
            prev_tensor = self.output
            n_in = prev_tensor.shape.as_list()[-1]
            for (f, n_c) in zip(conv_filter_sizes, num_conv_filters):
                if (f != 1):
                    if shield_channels == True:
                        n_s = min(n_c, (f**2)*n_in)//2
                        self.add_conv_layer(1, n_s, 1, activation='none')
                        self.layer_num += -1
                    elif shield_channels == False:
                        pass
                    else:
                        self.add_conv_layer(1, shield_channels, 1, activation='none')
                        self.layer_num += 1
                    self.add_conv_layer(f, n_c, stride, activation='none', name=name+'_conv')
                    self.layer_num += -1
                else:
                    self.add_conv_layer(f, n_c, stride, activation='none', name=name+'_conv')
                    self.layer_num +=-1
                outputs.append(self.output)
                self.output = prev_tensor
            for (f, n_c) in zip(max_pool_filter_sizes, num_max_pool_filters):
                self.add_max_pool_layer(f, stride, name=name+'_max_pool', activation='none')
                self.layer_num += -1
                self.add_conv_layer(1, n_c, 1, activation='none')
                self.layer_num += -1
                outputs.append(self.output)
                self.output = prev_tensor
            A = tf.concat(outputs, axis=-1)/(len(conv_filter_sizes) + len(max_pool_filter_sizes))
            name = name+'_block_'+str(self.layer_num)
            self.output = self.activation(A, activation, name)
            self.layer_num += 1
    
    def start_skip_connection(self):
        assert self.skip_start is None, 'already have root tensor of skip connection'
        with self.G.as_default():
            self.skip_start = self.output
    
    def end_skip_connection(self, activation='relu', name='residual'):
        assert self.skip_start is not None, 'haven\'t specified root tensor of skip connection'
        with self.G.as_default():
            self.output = self.activation(self.output + self.skip_start, activation=activation, name=name+str(self.layer_num))
            self.skip_start = None
    
    def add_loss_function(self, loss_type='xentropy'):
        with self.G.as_default():
            self.labels = tf.placeholder(dtype=tf.float32, shape=self.output.shape.as_list(), name='labels')
            if loss_type == 'xentropy':
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels), name='loss')
            elif loss_type == 'mean_square_error':
                self.loss = tf.reduce_mean(tf.square(self.output - self.labels), name='loss')
            else:
                raise Exception('unknown loss function')
    
    def define_training_op(self, optimizer):
        with self.G.as_default():
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            if optimizer.lower() == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif optimizer.lower() == 'sgd':
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif optimizer.lower() == 'rms':
                opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            else:
                raise Exception('unknown optimizer')
            self.training_op = opt.minimize(self.loss, name='training_op')
    
    def save_graph(self, save_path):
        with self.G.as_default():
            saver = tf.train.Saver(var_list=None)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, save_path)


a = ResNet(tf.float32, shape=[None,32,32,3]) # 32x32x3
a.add_inception_block([1,3,5,7], [8,8,8,8], 1, shield_channels=False) # 32x32x32
a.add_inception_block([1,3,5,7], [8,8,8,8], 1, shield_channels=False)
a.add_max_pool_layer(2,2) # 16x16x32
a.add_inception_block([1,3,5], [16,16,16], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[16]) # 16x16x64
a.start_skip_connection()
a.add_inception_block([1,3,5], [16,16,16], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[16])
a.add_inception_block([1,3,5], [16,16,16], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[16], activation='none')
a.end_skip_connection()
a.start_skip_connection()
a.add_inception_block([1,3,5], [16,16,16], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[16])
a.add_inception_block([1,3,5], [16,16,16], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[16], activation='none')
a.end_skip_connection()
a.add_max_pool_layer(2,2) # 8x8x64
a.add_inception_block([1,3,5], [32,32,32], 1, max_pool_filter_sizes=[2], num_max_pool_filters=[32]) #8x8x128
a.start_skip_connection()
a.add_inception_block([1,3,5], [32,32,32], 1, max_pool_filter_sizes=[2], num_max_pool_filters=[32])
a.add_inception_block([1,3,5], [32,32,32], 1, max_pool_filter_sizes=[2], num_max_pool_filters=[32], activation='none')
a.end_skip_connection()
a.start_skip_connection()
a.add_inception_block([1,3,5], [32,32,32], 1, max_pool_filter_sizes=[2], num_max_pool_filters=[32])
a.add_inception_block([1,3,5], [32,32,32], 1, max_pool_filter_sizes=[2], num_max_pool_filters=[32], activation='none')
a.end_skip_connection()
a.add_max_pool_layer(2,2) # 4x4x128
a.start_skip_connection()
a.add_inception_block([1,2,3], [32,32,32], 1, max_pool_filter_sizes=[2], num_max_pool_filters=[32])
a.add_inception_block([1,2,3], [32,32,32], 1, max_pool_filter_sizes=[2], num_max_pool_filters=[32], activation='none')
a.end_skip_connection()
a.add_conv_layer(1, 16, 1)
a.flatten_layer()
a.add_dense_layer(256)
a.add_dense_layer(256)
a.add_output_layer(10, activation='none')
a.add_loss_function('xentropy')
a.define_training_op('adam')
a.save_graph('./checkpoints/CIFAR10_0')

'''
# Test to make sure everything works out and time the forward/backward propagation:
import numpy as np
import time
import utils as u

with a.get_graph().as_default():
    X = a.input
    Y = a.output
    loss = a.loss
    labels = a.labels
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tic = time.time()
        y, L, _ = sess.run([Y, loss, a.training_op], feed_dict={X:np.random.randn(100,32,32,3), a.labels:u.one_hot(np.random.randint(10, size=100), 10), a.learning_rate:1e-3})
        toc = time.time()
        print(y.shape)
        print('{} sec per forward/backward pass per example'.format((toc-tic)/100))
        print('mean: {}, std. dev: {}, loss: {}'.format(np.mean(y), np.std(y), L))
'''