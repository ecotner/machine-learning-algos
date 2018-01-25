# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:54:58 2018

Convenience class for building/training/predicting using an MLP

@author: Eric Cotner
"""

import tensorflow as tf
import numpy as np

class MLP(object):
    '''
    Class for building simple multi-layer perceptron (MLP). Nothing fancy, just a couple fully-connected layers stacked together.
    '''
    def __init__(self, input_size=None):
        import tensorflow as tf
        import numpy as np
        self.G = tf.Graph()
        with self.G.as_default():
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='input')
            self.output = self.input
            self.loss = None
            self.labels = None
            self.optimizer_placeholders = None
            self.train_op = None
    
    def f_act(self, X, activation):
        with self.G.as_default():
            if activation == 'relu':
                return tf.nn.relu(X)
            elif activation == 'tanh':
                return tf.nn.tanh(X)
            elif activation == 'sigmoid':
                return tf.nn.sigmoid(X)
            elif activation == 'softmax':
                return tf.nn.softmax(X)
            elif activation in ['none', None, 'identity', 'linear']:
                return X
            else:
                raise Exception('Unknown activation function')
    
    def apply_function(self, func):
        ''' Applies an arbitrary function to the current output. Useful for inserting a data augmentation routine or something. '''
        self.output = func(self.output)
    
    def add_layer(self, width, activation='relu'):
        with self.G.as_default():
            n_in = self.output.shape.as_list()[1]
            W = tf.Variable(tf.random_normal(shape=[n_in, width]))
            tf.add_to_collection('weights', W)
            b = tf.Variable(tf.zeros([width]))
            self.output = self.f_act(tf.add(tf.matmul(self.output, W), b), activation)
    
    def add_loss(self, loss_type, regularizer='none'):
        ''' Applies loss function and regularizer to the network. Also return a prediction tensor.'''
        with self.G.as_default():
            assert self.loss is None, 'Already applied loss function'
            self.labels = tf.placeholder(tf.float32, shape=self.output.shape, name='labels')
            # Implement standard loss
            if loss_type.lower() in ['xentropy', 'x-entropy', 'cross-entropy', 'crossentropy', 'cross entropy']:
                self.output = tf.nn.softmax(self.output)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels))
            elif loss_type.lower() in ['mean_square_error', 'mean square error', 'mse', 'mean_square', 'L2']:
                self.loss = tf.reduce_mean(tf.square(self.output - self.labels))
            else:
                raise Exception('Unknown loss function')
            self.output = tf.identity(self.output, name='output')
            # Implement regularization loss
            if regularizer == 'L1':
                n_W = 0
                reg_loss = tf.constant(0., dtype=tf.float32)
                for W in tf.get_collection('weights'):
                    n_W += np.prod(W.shape.as_list())
                    reg_loss += tf.reduce_sum(tf.abs(W))
                self.regularization_parameter = tf.placeholder(dtype=tf.float32, shape=[])
                self.loss = self.loss + (self.regularization_parameter/n_W)*reg_loss
            elif regularizer == 'L2':
                n_W = 0
                reg_loss = tf.constant(0., dtype=tf.float32)
                for W in tf.get_collection('weights'):
                    n_W += np.prod(W.shape.as_list())
                    reg_loss += tf.reduce_sum(tf.square(W))
                self.regularization_parameter = tf.placeholder(dtype=tf.float32, shape=[])
                self.loss = self.loss + (self.regularization_parameter/n_W)*reg_loss
            elif regularizer in ['none', 'None', None]:
                pass
            else:
                raise Exception('Unknown regularizer')
            self.loss = tf.identity(self.loss, name='loss')
    
    def add_optimizer(self, optimizer):
        with self.G.as_default():
            assert self.loss is not None, 'Need to add loss before optimizer'
            assert self.optimizer_placeholders is None, 'Already added optimizer'
            assert self.train_op is None, 'Already added optimizer'
            if optimizer.lower() == 'adam':
                learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
                beta1 = tf.placeholder_with_default(0.9, shape=[], name='beta1')
                beta2 = tf.placeholder_with_default(0.999, shape=[], name='beta2')
                tf.add_to_collection('optimizer_placeholders', learning_rate)
                tf.add_to_collection('optimizer_placeholders', beta1)
                tf.add_to_collection('optimizer_placeholders', beta2)
                self.optimizer_placeholders = [learning_rate, beta1, beta2]
                opt = tf.train.AdamOptimizer(learning_rate, beta1, beta2, name='optimizer')
            elif optimizer.lower() == 'sgd':
                learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
                tf.add_to_collection('optimizer_placeholders', learning_rate)
                self.optimizer_placeholders = [learning_rate]
                opt = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer')
            elif optimizer.lower() == 'momentum':
                learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
                momentum = tf.placeholder_with_default(0.999, shape=[], name='momentum')
                tf.add_to_collection('optimizer_placeholders', learning_rate)
                tf.add_to_collection('optimizer_placeholders', momentum)
                self.optimizer_placeholders = [learning_rate, momentum]
                opt = tf.train.MomentumOptimizer(learning_rate, momentum, name='optimizer')
            else:
                raise Exception('Unknown optimizer')
            self.train_op = opt.minimize(self.loss, name='train_op')
    
    def save_graph(self, save_path):
        with self.G.as_default():
            if len(save_path) > len('.meta'):
                if save_path[-5:] == '.meta':
                    tf.train.export_meta_graph(save_path)
                else:
                    tf.train.export_meta_graph(save_path+'.meta')
    
    def load_graph(self, save_path, clear_devices=False):
        self.G = tf.Graph()
        with self.G.as_default():
            if len(save_path) > len('.meta'):
                if save_path[-5:] == '.meta':
                    tf.train.import_meta_graph(save_path, clear_devices)
                else:
                    tf.train.import_meta_graph(save_path+'.meta', clear_devices)
            self.input = self.G.get_tensor_by_name('input:0')
            self.output = self.G.get_tensor_by_name('output:0')
            self.loss = self.G.get_tensor_by_name('loss:0')
            self.labels = self.G.get_tensor_by_name('labels:0')
            self.optimizer_placeholders = tf.get_collection('optimizer_placeholders')
            self.train_op = self.G.get_operation_by_name('train_op')
    
    def train(self, X_train, Y_train, batch_size=None, feed_dict_extra={}, restore_from_checkpoint=None):
        ''' Performs the training process. Assumes the data has already been preprocessed. '''
        with self.G.as_default():
            if restore_from_checkpoint is not None:
                pass
            else:
                
    
    def predict(self):
        pass










if __name__ == '__main__':
    model = MLP(5)
    model.add_layer(10)
    model.add_layer(5, activation='none')
    model.add_loss('xentropy', regularizer='L2')
    model.add_optimizer('momentum')
    model.save_graph('./test_graph.meta')
    model.load_graph('./test_graph')
    print(model.G.get_collection('optimizer_placeholders'))








