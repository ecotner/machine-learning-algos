# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:07:27 2018

@author: Eric Cotner
"""

def unpickle(file):
    ''' Unpickles a file and returns a dictionary of the enclosed data. '''
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_graph(save_path, clear_devices=False, import_scope=None, **kwargs):
    import tensorflow as tf
    G = tf.Graph()
    with G.as_default():
        saver = tf.train.import_meta_graph(save_path+'.meta', clear_devices, import_scope, **kwargs)
    return G

def one_hot(idx, depth):
    import numpy as np
    M = np.zeros([len(idx),depth])
    for i, j in enumerate(idx):
        M[i,j] = 1
    return M

