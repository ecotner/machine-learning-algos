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
            self.most_recent_save_path = None
            self.EPSILON = 1e-8
    
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
            elif activation == 'maxout':
                a1 = tf.Variable(0.2*tf.ones(shape=X.shape[1]))
                a2 = tf.Variable(1.0*tf.ones(shape=X.shape[1]))
                return tf.maximum(a1*X, a2*X)
            elif activation in ['none', None, 'identity', 'linear']:
                return tf.identity(X)
            else:
                raise Exception('Unknown activation function')
    
    def apply_function(self, func):
        ''' Applies an arbitrary function to the current output. Useful for inserting a data augmentation routine or something. '''
        self.output = func(self.output)
    
    def add_layer(self, width, activation='relu'):
        with self.G.as_default():
            n_in = self.output.shape.as_list()[1]
            W = tf.Variable(tf.random_normal(shape=[n_in, width])*np.sqrt(2/(n_in+width)))
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
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels))
                self.output = tf.nn.softmax(self.output)
            elif loss_type.lower() == 'sigmoid':
                self.loss = tf.reduce_mean(self.labels*tf.log(1+tf.exp(-self.output)+self.EPSILON) + (1-self.labels)*tf.log(1+tf.exp(self.output+)+self.EPSILON))
                self.output = tf.sigmoid(self.output)
            elif loss_type.lower() in ['mean_square_error', 'mean square error', 'mse', 'mean_square', 'L2']:
                self.loss = tf.reduce_mean(tf.square(self.output - self.labels))
            else:
                raise Exception('Unknown loss function')
            self.output = tf.identity(self.output, name='output')
            # Implement regularization loss
            self.regularization_parameter = tf.placeholder_with_default(0., shape=[], name='regularization_parameter')
            if regularizer == 'L1':
                n_W = 0
                reg_loss = tf.constant(0., dtype=tf.float32)
                for W in tf.get_collection('weights'):
                    n_W += np.prod(W.shape.as_list())
                    reg_loss += tf.reduce_sum(tf.abs(W))
                self.loss = self.loss + (self.regularization_parameter/n_W)*reg_loss
            elif regularizer == 'L2':
                n_W = 0
                reg_loss = tf.constant(0., dtype=tf.float32)
                for W in tf.get_collection('weights'):
                    n_W += np.prod(W.shape.as_list())
                    reg_loss += tf.reduce_sum(tf.square(W))
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
            # Create training operation and gradient clipping
            grads_and_vars = opt.compute_gradients(self.loss)
            grads, variables = zip(*grads_and_vars)
            grad_avgs = [tf.Variable(1000*tf.ones(shape=tf.shape(grad)), trainable=False) for grad in grads]
            [tf.assign(x[0], 0.9 * x[0] + 0.1 * x[1]) for x in zip(grad_avgs, grads)]
            avg_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(x)) for x in grad_avgs]))
            self.grads, _ = tf.clip_by_global_norm(grads, 10*avg_norm, name='grads')
            self.train_op = opt.apply_gradients(zip(self.grads, variables), name='train_op')
    
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
            self.grads = self.G.get_tensor_by_name('grads:0')
    
    def train(self, X_train, Y_train, batch_size=None, learning_rate=1e-3, max_epochs='dynamic', regularization_parameter=None, feed_dict_extra={}, save_path=None, verbosity='low'):
        ''' Performs the training process. Assumes the data has already been preprocessed. '''
        m_train = X_train.shape[0]
        lr = self.G.get_tensor_by_name('learning_rate:0')
        lambda_reg = self.G.get_tensor_by_name('regularization_parameter:0')
        if batch_size is None:
            batch_size = max(m_train//20, 1)
        if max_epochs == 'dynamic':
            max_epochs = int(1e9)
            dynamic_max_epochs = True
        else:
            dynamic_max_epochs = False
        if regularization_parameter is None:
            regularization_parameter = 0
        best_epoch = 1
        best_loss = np.inf
        
        with self.G.as_default():
            with tf.Session() as sess:
                saver = tf.train.Saver()
                if save_path is None:
                    sess.run(tf.global_variables_initializer())
                    save_path = './mlp'
                else:
                    assert type(save_path) == str, 'Checkpoint path not valid'
                    saver.restore(save_path)
                self.most_recent_save_path = save_path
                
                global_step = 0
                for epoch in range(max_epochs):
                    
                    for b in range(m_train//batch_size):
                        start_slice = b*batch_size
                        end_slice = min((b+1)*batch_size, m_train)
                        feed_dict = {**{self.input:X_train[start_slice:end_slice],
                                        self.labels:Y_train[start_slice:end_slice],
                                        lr:learning_rate,
                                        lambda_reg:regularization_parameter},
                                        **feed_dict_extra}
                        loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                        
                        if np.isnan(loss) or np.isinf(loss):
                            print('Detected nan/inf, ending training')
                            output = sess.run(self.output, feed_dict=feed_dict)
                            grads = sess.run(self.grads, feed_dict=feed_dict)
                            print('Output:')
                            print(output)
                            print('Grads:')
                            print(grads)
                            break
                        
                        if (epoch >= 1) and ((not np.isnan(loss)) or (not np.isinf(loss))):
                            if loss < best_loss:
                                best_loss = loss
                                best_epoch = epoch
                                saver.save(sess, save_path, write_meta_graph=False)
                        
                        if verbosity == 'high':
                            print('Epoch: {}, batch {}/{}, loss: {:.3e}'.format(epoch+1, b, m_train//batch_size, loss))
                        elif (type(verbosity) != str):
                            if global_step % verbosity == 0:
                                print('Epoch: {}, batch {}/{}, loss: {:.3e}'.format(epoch+1, b, m_train//batch_size, loss))
                        
                        global_step += 1
                    
                    # Do stuff at the end of the epoch
                    if np.isnan(loss) or np.isinf(loss):
                            break
                    
                    if verbosity in ['high', 'low']:
                        print('Epoch: {}, loss: {:.3e}'.format(epoch+1, loss))
                    
                    if dynamic_max_epochs and epoch >= 10:
                        if epoch/best_epoch > 1.2:
                            break
                
                print('Training complete')
    
    def predict(self, X, save_path=None):
        with self.G.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                if save_path is None:
                    saver.restore(sess, save_path=self.most_recent_save_path)
                else:
                    saver.restore(sess, save_path)
                    self.most_recent_save_path = save_path
                output = sess.run(self.output, feed_dict={self.input:X})
        return output










if __name__ == '__main__':
    model = MLP(5)
    model.add_layer(10, activation='maxout')
    model.add_layer(1, activation='none')
    model.add_loss('sigmoid', regularizer='L2')
    model.add_optimizer('adam')
#    model.save_graph('./derp')
#    model.load_graph('./derp')
    model.train(np.random.randn(100000,5), np.random.choice([0,1], size=[100000,1]), learning_rate=1e-2, verbosity='high')
    pred = model.predict(np.random.randn(100000,5))
    print(np.all(pred==1))








