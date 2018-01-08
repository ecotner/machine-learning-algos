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
        X: the input tensor
        Y: the (most recent) output tensor
        skip_start = keeps track of the node at which the skip connection starts
        layer_num: the layer index of the (most_recent) output tensor (layer indexing starts at 1, and skip connections and flattening operations don't count towards it)
    
    Methods:
        get_graph(): returns the computational graph ResNet.G
        get_tensor(): returns tensor from the computational graph by name
        reset_graph: resets the graph to a blank state
        add_conv_layer: adds a convolutional layer to the most recent output
        add_dense_layer: adds a dense/fully-connected layer to the most recent output
        add_max_pool_layer: adds a max pooling layer to the most recent output
        flatten_layer flattens the most recent output to a form appropriate for attaching dense layers
        add_inception_block: adds an inception block composed of multiple convolutional filters
        start_skip_connection: marks the root node of a skip connection
        end_skip_connection: adds the most recent output layer to the layer marked by the start_skip_connection method
    '''
    
    def __init__(self, dtype, shape, name='X'):
        ''' Initialize the graph with a single input layer. '''
        self.G = tf.Graph()
        with self.G.as_default():
            self.X = tf.placeholder(dtype, shape, name)
            self.Y = self.X
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
    
    def add_conv_layer(self, f, n_c, s, padding='SAME', use_cudnn_on_gpu=None, data_format=None, name='conv'):
        '''
        Adds a convolutional layer to the network. All filter weights have Xavier initialization, and biases are initialized to zeros.
        Arguments:
            f: the dimension of the filter (expects square input)
            n_c: the number of output channels (or number of filters to be applied)
            s: the stride
            padding: whether to use 'SAME' or 'VALID' padding
            use_cudnn_on_gpu: whether to use cuDNN on your GPU
            data_format: specifies the data format of the input/output channels. Default is 'NHWC', with alternative option 'NCHW'
            name: an optional name for the convolutional layer (the layer index will be appended to this)
        '''
        with self.G.as_default():
            n_in = self.Y.shape.as_list()[-1]
            W = tf.Variable(initial_value=tf.random_normal(shape=[f,f,n_in,n_c], dtype=tf.float32)*2/tf.sqrt(tf.cast(f*f*n_in + n_c, tf.float32)), dtype=tf.float32, name='W_'+name+'_{0}x{0}_{1}'.format(f, self.layer_num))
            b = tf.Variable(initial_value=tf.zeros(shape=[n_c]), dtype=tf.float32, name='b_'+name+'_{0}x{0}_{1}'.format(f, self.layer_num))
            self.Y = tf.nn.relu(tf.nn.conv2d(self.Y, W, [1,s,s,1], padding, use_cudnn_on_gpu=None, data_format=None) + b, name=name+'_{0}x{0}_{1}'.format(f, self.layer_num))
            self.layer_num += 1
    
    def add_dense_layer(self, width, name='dense'):
        '''
        Adds a dense/fully-connected layer to the network. Weights have Xavier initialization and biases are initialized to zeros.
        Arguments:
            width: the number of neurons in the network
            name: an optional name for the layer (the layer index will be appended to this)
        '''
        with self.G.as_default():
            n_in = self.Y.shape.as_list()[-1]
            W = tf.Variable(initial_value=tf.random_normal(shape=[n_in,width], dtype=tf.float32)*2/tf.sqrt(tf.cast(n_in + width, tf.float32)), name='W_'+name+str(self.layer_num))
            b = tf.Variable(initial_value=tf.zeros(shape=[width]), dtype=tf.float32, name='b_'+name+str(self.layer_num))
            self.Y = tf.nn.relu(tf.matmul(self.Y, W) + b, name=name+str(self.layer_num))
            self.layer_num += 1
    
    def add_output_layer(self, width, name='Y'):
        with self.G.as_default():
            n_in = self.Y.shape.as_list()[-1]
            W = tf.Variable(initial_value=tf.random_normal(shape=[n_in,width], dtype=tf.float32)*2/tf.sqrt(tf.cast(n_in + width, tf.float32)), name='W_'+name+str(self.layer_num))
            b = tf.Variable(initial_value=tf.zeros(shape=[width]), dtype=tf.float32, name='b_'+name+str(self.layer_num))
            self.Y = tf.add(tf.matmul(self.Y, W), b, name=name)
            self.layer_num += 1

    def add_max_pool_layer(self, f, s, name='max_pool'):
        '''
        Adds a max pool layer to the network.
        Arguments:
            f: dimensions of square filter
            s: stride
            name: an optional name (the layer index will be appended to this)
        '''
        with self.G.as_default():
            self.Y = tf.nn.max_pool(self.Y, [1,f,f,1], [1,s,s,1], padding='SAME', name=name+'_{0}x{0}_{1}'.format(f, self.layer_num))
            self.layer_num += 1
    
    def flatten_layer(self):
        ''' Flattens the output of a previous covolutional layer so that fully connected layers can be attached. '''
        with self.G.as_default():
            self.Y = tf.contrib.layers.flatten(self.Y)

    def add_inception_block(self, conv_filter_sizes, num_conv_filters, stride=1, max_pool_filter_sizes=[], num_max_pool_filters=[], name='inception'):
        '''
        Creates an inception block composed of multiple convolutional filters and max pool layers, concatenated into a single tensor. In order to reduce computation, each convolutional filter larger than 1x1 is preceded by a 1x1 convolutional filter with output channels equal to 1/2 of the specified output channels of that convolution.
        Arguments:
            conv_filter_sizes: a list of dimensions of square convolutional filters
            num_conv_fiters: the number of filters of each type to be applied
            stride: the (shared) stride of each filter (all strides must be the same so the concatenation operation is well-defined)
            max_pool_filter_sizes: list of dimensions of square max pool filters
            num_max_pool_filters: the number of filters coming from the max pool channel (a 1x1 convolution is applied after max pool to change the channel dimension)
            name: an optional name for the inception block
        '''
        assert all([(f>0) for f in conv_filter_sizes]), 'some conv_filter_sizes are negative'
        with self.G.as_default():
            outputs = []
            prev_tensor = self.Y
            for (f, n_c) in zip(conv_filter_sizes, num_conv_filters):
                if (f != 1):
                    self.add_conv_layer(1, n_c//2, 1)
                    self.layer_num += -1
                    self.add_conv_layer(f, n_c, stride, name=name+'_conv')
                    self.layer_num += -1
                else:
                    self.add_conv_layer(f, n_c, stride, name=name+'_conv')
                    self.layer_num +=-1
                outputs.append(self.Y)
                self.Y = prev_tensor
            for (f, n_c) in zip(max_pool_filter_sizes, num_max_pool_filters):
                self.add_max_pool_layer(f, stride, name=name+'_max_pool')
                self.layer_num += -1
                self.add_conv_layer(1, n_c, 1)
                self.layer_num += -1
                outputs.append(self.Y)
                self.Y = prev_tensor
            self.Y = tf.nn.relu(tf.concat(outputs, axis=-1), name=name+'_block_'+str(self.layer_num))
            self.layer_num += 1
    
    def start_skip_connection(self):
        assert self.skip_start is None, 'already have root tensor of skip connection'
        self.skip_start = self.Y
    
    def end_skip_connection(self, name='residual'):
        assert self.skip_start is not None, 'haven\'t specified root tensor of skip connection'
        self.Y = tf.nn.relu(self.Y + self.skip_start, name=name+str(self.layer_num))
        self.skip_start = None

'''
# Testing the construction:
import numpy as np

a = ResNet(tf.float32, shape=[None,60,60,3])
a.add_conv_layer(5,16,2)
a.add_max_pool_layer(2,2)
a.start_skip_connection()
a.add_inception_block([1,3,5], [8,8,8], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[8])
a.add_inception_block([1,3,5], [4,4,4], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[4])
a.end_skip_connection()
a.add_max_pool_layer(2,2)
a.start_skip_connection()
a.add_inception_block([1,3,5], [8,8,8], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[8])
a.add_inception_block([1,3,5], [4,4,4], 1, max_pool_filter_sizes=[3], num_max_pool_filters=[4])
a.end_skip_connection()
a.flatten_layer()
a.add_dense_layer(256)
a.add_dense_layer(256)
a.add_output_layer(10)

with a.get_graph().as_default():
    X = a.X
    Y = a.Y
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(Y, feed_dict={X:np.random.randn(10,60,60,3)})
        print(y.shape)
        print(np.mean(y, axis=0))
'''