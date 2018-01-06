#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:54:03 2017

Collection of utilities for use in playing pong

@author: ecotner
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import collections

# Image preprocessing
def preprocess_image(image, d=2):
    '''
    Takes in a numpy array of size 210x160x3, representing an RGB image, and returns a cropped greyscale image of size (160x160)//d with boolean pixel intensities (to compress data).
    '''
    x = image[34:194,:,1]
    x = np.where(x>100, np.ones(x.shape, dtype=bool), np.zeros(x.shape, dtype=bool))
    X = np.zeros((160//d,160//d), dtype=bool)
    for i in range(160//d):
        for j in range(160//d):
            X[i,j] = np.any(x[d*i:d*(i+1),d*j:d*(j+1)])
    return X.astype(bool)

# Progress monitor
def print_update(n, N, period, message='Progress: ', tic=None, toc=None):
    ''' Periodically prints a progress update with a given period. Assumes the function is only being called once per period, and the program is being run in a UNIX terminal to move the cursor correctly. '''
    progress = n/N
    n_blocks = int(50*progress)
    progress_str = '['
    for block in range(n_blocks):
        progress_str += '#'
    for block in range(50-n_blocks):
        progress_str += '-'
    progress_str += ']'
    print('\033[K{}{}/{}'.format(message,n,N), end='', flush=True)
    if (tic is not None) and (toc is not None):
        dt = toc - tic
        rate = period/dt
        time_left = int((N-n)/rate)
        hours = time_left//3600
        minutes = (time_left % 3600)//60
        seconds = (time_left % 60)
        print(', rate: {:.2e}/s, est. time left: {:02d}:{:02d}:{:02d}'.format(rate, hours, minutes, seconds), end='', flush=True)
    print('\n\033[K'+progress_str+'\t{:.1f}% complete'.format(100*progress), end='')
    if n != N:
        print('\033[1A\r', end='')
    else:
        print('', end='\n')

# Gather pong screens for validation purposes
def collect_pong_screens(max_episodes, steps_to_skip=1, max_to_keep=250, ds_factor=2):
    # Initialize gym environment
    env = gym.make('Pong-v0')
    # Iterate over episodes
    replay_memory = ReplayMemory(max_exp_len=max_to_keep, max_frame_len=4)
    for episode in range(max_episodes):
        obs = preprocess_image(env.reset())
        for i in range(3):
            replay_memory.add_frame(np.zeros((160//ds_factor,160//ds_factor)))
        replay_memory.add_frame(obs)
    
        # Iterate over frames
        done = False
        while not done:
            # Decide on action
            a = np.random.choice([1,2,3])
            
            # Take action, observe environment, reward
            obs, r, done, _ = env.step(a)
            r_sum = r
            for i in range(steps_to_skip):
                obs, r, done_temp, _ = env.step(1)
                r_sum += r
                if done_temp == True:
                    done = True
            
            # Add new state/reward to replay memory
            replay_memory.add_frame(preprocess_image(obs))
            experience = np.stack(list(replay_memory.frames), axis=-1)
            replay_memory.add_exp(experience)
            
    print('Number of frames collected: {}'.format(len(replay_memory.experiences)))
    # Save frames to disk
    print('Saving frames to disk...')
    np.save('./val_Pong_frames.npy', np.stack(list(replay_memory.experiences), axis=0))
    print('Frames saved!')

# Load pong screens
def load_validation_screens():
    X_val = np.load('./val_Pong_frames.npy')
    return X_val

# Load preexisting graph
def load_graph(save_path, is_training=True):
    ''' Loads the graph and weights from a .meta file. '''
    if is_training:
        device = '/gpu:0'
    else:
        device = '/cpu:0'
    with tf.device(device):
        G = tf.Graph()
        with G.as_default():
            saver = tf.train.import_meta_graph(save_path + '.meta', clear_devices=(not is_training))
            # Restore Variables
            with tf.Session() as sess:
                saver.restore(sess, save_path)
    return G

class ReplayMemory(object):
    ''' Containins all the necessary elements for the replay memory. Keeps a running list of the current frames, past experiences, and operations on the two. The experiences are organized into a deque of tuples with elements (frames, action, reward, done), and the frames attribute is a length 5 deque containing the most recent frames encountered. '''
    def __init__(self, max_exp_len=1000, max_frame_len=5):
        self.experiences = collections.deque(maxlen=max_exp_len)
        self.frames = collections.deque(maxlen=max_frame_len)
    
    def add_frame(self, frame):
        self.frames.append(frame)
    
    def add_exp(self, exp):
        self.experiences.append(exp)
    
    def sample(self, batch_size):
        mem_len = len(self.experiences)
        batch_len = min(batch_size, mem_len)
        idx = np.random.permutation(mem_len)[:batch_len]
        state_list = []
        action_list = []
        reward_list = []
        done_list = []
        for i in idx:
            exp = self.experiences[i]
            state_list.append(exp[0])
            action_list.append(exp[1])
            reward_list.append(exp[2])
            done_list.append(exp[3])
        states = np.stack(state_list, axis=0)
        actions = np.stack(action_list, axis=0)
        rewards = np.stack(reward_list, axis=0)
        done = np.stack(done_list, axis=0)
        return states, actions, rewards, done

class ReplayMemory1(object):
    ''' Containins all the necessary elements for the replay memory. Keeps a running list of the current frames, past experiences, and operations on the two. The experiences are organized into a set of tuples with elements (frames, action, reward, done), and the frames attribute is a length 5 deque containing the most recent frames encountered. Random access of this memory should be very efficient since it is a set, at the cost of the elements being unordered, so that removal of experiences once the memory is full will be stochastic rather than FIFO. However the odds are favorable that each experience will be sampled many times before being discarded. '''
    def __init__(self, max_exp_len=1000, max_frame_len=5):
        self.max_exp_len = max_exp_len
        self.max_frame_len = max_frame_len
        self.experiences = set()
        self.frames = collections.deque(maxlen=self.max_frame_len)
    
    def add_frame(self, frame):
        self.frames.append(frame)
    
    def add_exp(self, exp):
        self.experiences.add(exp)
        if len(self.experiences) > self.max_exp_len:
            self.experiences.pop()
    
    def sample(self, batch_size):
        mem_len = len(self.experiences)
        batch_len = min(batch_size, mem_len)
        state_list = []
        action_list = []
        reward_list = []
        done_list = []
        for i in range(batch_len):
            exp = self.experiences.pop()
            state_list.append(exp[0])
            action_list.append(exp[1])
            reward_list.append(exp[2])
            done_list.append(exp[3])
            self.experiences.add(exp)
        states = np.stack(state_list, axis=0)
        actions = np.stack(action_list, axis=0)
        rewards = np.stack(reward_list, axis=0)
        done = np.stack(done_list, axis=0)
        return states, actions, rewards, done
            
def plot_metrics(X, name, title, x_label, y_label, *Y):
    '''
    Plots multiple quantities on the same plot
    '''
    plt.figure(name)
    plt.clf()
    for y in Y:
        array, label = y
        plt.plot(X, array, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(name+'.png', bbox_inches='tight')
        

# Testing stuff
def test_play(steps_to_skip=0):
    env = gym.make('Pong-v0')
    obs = env.reset()
    for i in range(30):
        obs = env.step(1)
    a = 1
    while True:
        for i in range(steps_to_skip):
            obs, reward, done, _ = env.step(1)
        obs, reward, done, _ = env.step(int(a))
        s = preprocess_image(obs, d=2)
        print('min: {}, max: {}, agv: {}'.format(np.min(s), np.max(s), np.mean(s)))
        plt.imshow(s, cmap='gray')
        plt.show()
        a = input('Enter action [1,2,3] to continue, or q to quit: ')
        if a == 'q':
            break
    
#test_play(1)
































