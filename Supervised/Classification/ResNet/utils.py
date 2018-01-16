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
        tf.train.import_meta_graph(save_path+'.meta', clear_devices, import_scope, **kwargs)
    return G

def one_hot(idx, depth):
    import numpy as np
    M = np.zeros([len(idx),depth])
    for i, j in enumerate(idx):
        M[i,j] = 1
    return M

def input_choice(query_str, choices=['y','n','']):
    ''' Streamlines asking a question from the user by automatically rejecting an answer if it isn't one of the offered choices. Default options are "y", "n", and "". '''
    done = False
    while not done:
        x = input(query_str)
        if x in choices:
            done = True
        else:
            print('Invalid choice!\n')
    return x

def random_crop(tensor, size):
    '''
    Does a random crop on a batch of images, assuming NHWC format.
    Arguments:
        tensor: the input tensor to be cropped. Assumes NHWC format.
        size: a list representing the output size of the crop with dimensions [N,H,W]
    Returns:
        cropped_tensor: the randomly cropped tensor
    '''
    import tensorflow as tf
    N, H, W, C = tensor.get_shape()
    dH = tf.random_uniform([], maxval=H-size[0]+1, dtype=tf.int32)
    dW = tf.random_uniform([], maxval=W-size[1]+1, dtype=tf.int32)
    dC = tf.random_uniform([], maxval=C-size[2]+1, dtype=tf.int32)
    cropped_tensor = tensor[:,dH:dH+size[0],dW:dW+size[1],dC:dC+size[2]]
    return cropped_tensor

def plot_metrics(data_path, plot_path=None):
    '''
    Plots the training/validation metrics collected in log files.
    '''
    # Import necessary modules
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    import numpy as np
    import re
    
    if plot_path is None:
        plot_path = data_path
    
    # Get training size, batch size, logging frequency
    with open(data_path+'.log', 'r') as fo:
        p_train = re.compile('Training data')
        p_val_batch = re.compile('Validation set size')
        p_train_batch = re.compile('Batch size')
        p_log = re.compile('Logging frequency')
        p_num = re.compile(r'\d+')
        for line in fo:
            if re.match(p_train, line) is not None:
                m = re.search(p_num, line)
                m_train = int(m.group())
            elif re.match(p_train_batch, line) is not None:
                m = re.search(p_num, line)
                b_train = int(m.group())
            elif re.match(p_val_batch, line) is not None:
                m = re.search(p_num, line)
                b_val = int(m.group())
            elif re.match(p_log, line) is not None:
                m = re.search(p_num, line)
                log_freq = int(m.group())
    
    # Plot the loss
    loss_dict = [('Training','_train_loss.log'), ('Validation','_val_loss.log')]
    plt.figure(num='Loss')
    plt.clf()
    ax = plt.gca()
    for name, file in loss_dict:
        loss_list = []
        with open(data_path+file, 'r') as fo:
            for line in fo:
                loss_list.append(float(line))
        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(loss_list))
        if name == 'Training':
            x_train = len(x)
        else:
            x = (x_train/len(x))*x
        plt.plot(x, loss_list, 'o', label=name, alpha=0.25)
    del loss_list
    plt.title('Average batch loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig(plot_path+'_loss.png')
    
    # Plot the error
    acc_dict = [('Training','_train_accuracy.log'), ('Validation','_val_accuracy.log')]
    plt.figure(num='Error')
    plt.clf()
    ax = plt.gca()
    for name, file in acc_dict:
        acc_list = []
        with open(data_path+file, 'r') as fo:
            for line in fo:
                acc_list.append(float(line))
        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(acc_list))
        if name == 'Training':
            x_train = len(x)
        else:
            x = (x_train/len(x))*x
        plt.plot(x, 100*(1-np.array(acc_list)), 'o' , label=name, alpha=0.25)
    del acc_list
    plt.title('Average batch error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig(plot_path+'_error.png')

if __name__ == '__main__':
    plot_metrics('./checkpoints/{0}/CIFAR10_{0}'.format(10))
