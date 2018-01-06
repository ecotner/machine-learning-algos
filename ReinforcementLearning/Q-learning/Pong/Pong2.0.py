#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:50:55 2017

Deep Q network trained to play Pong.

This network is 

@author: ecotner
"""

import numpy as np
import utils as u
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import time

IS_TRAINING = True
RELOAD_PARAMETERS = True
MAX_EPISODES = int(1e10)
REPLAY_MEM_LEN = int(5e4)
OBSERVE_STEPS = int(5e3)
PLOT_EVERY_N_STEPS = 100
STEPS_TO_SKIP = 1
MAX_EPSILON = 1
MIN_EPSILON = 0.05
EPSILON_ANNEALING_STEPS = int(1e6)
LEARNING_RATE = 1e-6
GAMMA = np.exp(-1/100)
BATCH_SIZE = 128
SAVE_PATH = './Checkpoints/6/DQN'
DOWNSAMPLE = 2
SEED = 0

def build_Q_network(conv_layers, fc_layers, activation='relu'):
    '''
    Constructs the deep Q network (DQN) used as a function approximator for the action value function Q(s,a). The network consists of several convolutional layers (with 2x2 max pooling applied after each), which are then flattened and passed through several fully connected layers. The final output is to three neurons, each corresponding to an action taken by the pong agent (stay,up,down).
    Input:
        conv_layers: a list of tuples (filter_dims,stride), where each tuple is a set of hyperparameters for a given layer, defined as:
            filter_dims: dimensions of the filter
            stride: the stride of the convolutional filter
        fc_layers: a list of tuples (width,dropout), where each tuple is a set of hyperparameters for a given layer, defined as:
            width: the number of neurons in this layer
            dropout: the fraction of neurons kept during dropout (0<dropout<=1)
    '''
    # Reset/initialize tensorflow graph
    tf.reset_default_graph()
    G = tf.Graph()
    # Build the layers
    with G.as_default():
        
        # Determine activation function
        if activation.lower() == 'relu':
            def f_act(x, name=None):
                return tf.nn.relu(x, name=name)
        elif activation.lower() == 'lrelu':
            def f_act(x, name=None):
                return tf.maximum(0.05*x, x, name=name)
        elif activation.lower() == 'sigmoid':
            def f_act(x, name=None):
                return tf.sigmoid(x, name=name)
        else:
            raise Exception('No known activation type')
        
        # Input layer
        X = tf.placeholder(dtype=tf.float32, shape=[None,160//DOWNSAMPLE,160//DOWNSAMPLE,4], name='X')
        A = tf.cast(X, tf.float32) # Input is saved as boolean array so need to convert to float
        
        # Convolutional layers
        for l, conv_layer in enumerate(conv_layers):
            filter_dims = conv_layer[0]
            stride = conv_layer[1]
            W = tf.Variable(initial_value=tf.random_normal(filter_dims, stddev=np.sqrt(2/(np.prod(filter_dims[:3]) + filter_dims[3]))), dtype=tf.float32, name='W'+str(l))
            b = tf.Variable(initial_value=.1*np.ones((filter_dims[-1],)), dtype=tf.float32, name='b'+str(l))
            conv = f_act(tf.nn.conv2d(A, W, strides=stride, padding='SAME') + b, name='conv'+str(l))
            A = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool'+str(l))
        
        # Flatten for FC layers
        A = tf.contrib.layers.flatten(A)
        
        # Fully connected layers
        for l, fc_layer in enumerate(fc_layers):
            width = fc_layer[0]
            dropout = fc_layer[1]
            prev_width = A.get_shape()[-1].value
            W = tf.Variable(initial_value=tf.random_normal((prev_width,width), stddev=np.sqrt(2/(prev_width + width))), dtype=tf.float32, name='W'+str(l+len(conv_layers)))
            b = tf.Variable(initial_value=.1*np.ones((width,)), dtype=tf.float32, name='b'+str(l+len(conv_layers)))
            Z = tf.matmul(A, W) + b
            A = tf.nn.dropout(f_act(Z), keep_prob=dropout, name='A'+str(l+len(conv_layers)))
        
        # Output layer (only 3 neurons)
        prev_width = A.get_shape()[-1].value
        W = tf.Variable(initial_value=tf.random_normal((prev_width,3), stddev=np.sqrt(2/(prev_width + 3))), dtype=tf.float32, name='W'+str(len(conv_layers)+len(fc_layers)))
        b = tf.Variable(initial_value=-1.3*np.ones((3,)), dtype=tf.float32, name='b'+str(len(conv_layers)+len(fc_layers)))
        tf.add(tf.matmul(A, W), b, name='Y')
    
    # Return the computational graph
    return G

# Train Q-network
def train(G, max_episodes, save_path):
    '''
    Trains a DQN to play pong.
    '''
    
    # Define some constants, lists, metrics, etc
    action_map = {1:'x', 2:'^', 3:'v'} # Stay, up, down
    replay_memory = u.ReplayMemory(max_exp_len=REPLAY_MEM_LEN)
    step_list = []
    reward_list = []
    val_Q_list = []
    episode_length_list = []
    avg_episode_length_list = []
    episode_score_list = {'player':[], 'computer':[]}
    X_val = u.load_validation_screens()
    
    # Initialize the Pong gym environment, set seeds
    env = gym.make('Pong-v0')
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    plt.ioff()
    
    # Gather screens
    
    # Initialize computational graph
    with G.as_default():
        # Get input/output tensors
        X = G.get_tensor_by_name('X:0')
        Y = G.get_tensor_by_name('Y:0')
        
        # Append loss function to graph
        Q = tf.placeholder(dtype=tf.float32, shape=[None], name='Q')
        A = tf.placeholder(dtype=tf.int32, shape=[None], name='A')
        mask = tf.one_hot(A, depth=3, dtype=tf.float32, axis=-1)
        L = tf.reduce_mean(tf.square(tf.reduce_sum(mask*Y, axis=-1) - Q), name='L')
        
        # Define optimizer, training op, gradient clipping, etc.
        if not RELOAD_PARAMETERS:
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE, name='Adam')
        else:
            optimizer = G.get_operation_by_name('Adam')
        saver = tf.train.Saver()
        # Initialize TF session
        with tf.Session() as sess:
            # Reload/initialize variables
            if RELOAD_PARAMETERS:
                print('Reloading from last checkpoint...')
                saver.restore(sess, save_path)
            else:
                print('Initializing variables...')
            gradients, variables = zip(*optimizer.compute_gradients(L))
            train_op = optimizer.apply_gradients(zip(gradients, variables))
            sess.run(tf.global_variables_initializer())
            # Iterate over episodes
            global_steps = 0
            for episode in range(max_episodes):
                obs = u.preprocess_image(env.reset())
                for i in range(3):
                    replay_memory.add_frame(np.zeros((160//DOWNSAMPLE,160//DOWNSAMPLE)))
                replay_memory.add_frame(obs)
            
                # Iterate over frames
                done = False
                frame = 0
                episode_score = [0,0]
                while not done:
                    if (global_steps >= OBSERVE_STEPS):
                        # Feed state into DQN
                        s = np.stack([replay_memory.frames[i] for i in range(-4,0)], axis=-1).reshape(1,160//DOWNSAMPLE,160//DOWNSAMPLE,4)
                        y = sess.run(Y, feed_dict={X:s})
                        
                        # Decide on action
                        epsilon = max(MAX_EPSILON*(1-global_steps/EPSILON_ANNEALING_STEPS), MIN_EPSILON)
                        if (np.random.rand() < epsilon):
                            a = np.random.choice([1,2,3])
                        else:
                            a = np.argmax(y)+1
                    else:
                        a = np.random.choice([1,2,3])
                    
                    # Take action, observe environment, reward
                    obs, r, done, _ = env.step(a)
                    r_sum = r
                    for i in range(STEPS_TO_SKIP):
                        obs, r, done_temp, _ = env.step(1)
                        r_sum += r
                        if done_temp == True:
                            done = True
                    if r_sum > 0:
                        episode_score[0] += r_sum
                    elif r_sum < 0:
                        episode_score[1] -= r_sum
                    
                    # Add new state/reward to replay memory
                    replay_memory.add_frame(u.preprocess_image(obs))
                    experience = (np.stack(list(replay_memory.frames), axis=-1), a, r_sum, done)
                    replay_memory.add_exp(experience)
                    
                    # Do training batch update
                    if (global_steps >= OBSERVE_STEPS):
                        S, A_, R, D = replay_memory.sample(BATCH_SIZE)
                        y2 = sess.run(Y, feed_dict={X:S[:,:,:,-4:]})
                        q = R + (1-D)*GAMMA*np.max(y2, axis=1)
                        _, batch_loss = sess.run([train_op, L], feed_dict={X:S[:,:,:,-5:-1], Q:q, A:(A_-1)})
                        if (batch_loss == np.nan):
                            print('nan error, exiting training')
                            exit()
                        elif (np.mean(np.max(y2, axis=-1)) > 1e2):
                            print('unstable Q value, exiting training')
                            exit()
                    
                        # Print updates
                        print('Episode: {}/{},\tframe: {},\treward: {},\t<max(Q)>: {:.3e},\nmax(Q): {:.3e},\taction: {},\tcurrent std(Q)/mean(Q): {:.3e}'.format(episode+1, max_episodes, frame+1, int(r_sum), np.mean(np.max(y2, axis=-1)), np.max(y), action_map[a], np.std(y)/np.mean(y)))
                        
                        # Plot frame-by-frame metrics
                        if global_steps == 0:
                            avg_reward = r_sum
                        else:
                            avg_reward = (1-np.exp(-1/500))*r_sum + np.exp(-1/500)*avg_reward
                        if (global_steps % PLOT_EVERY_N_STEPS == 0):
                            step_list.append(global_steps)
                            reward_list.append(10*avg_reward)
                            y_val = sess.run(Y, feed_dict={X:X_val})
                            val_Q_list.append(np.mean(np.max(y_val, axis=-1)))
                            u.plot_metrics(step_list, 'PongMetrics', 'Pong Metrics', 'Global step', '', (val_Q_list,'Validation <max(Q)>'), (reward_list, '10*<R>'))
                    else:
                        print('Observation step {}/{}'.format(global_steps, OBSERVE_STEPS))
                    
                    # Update state variables
                    global_steps += 1
                    frame += 1
                    
                # Save parameters at end of episode, plot episode metrics
                saver.save(sess, SAVE_PATH)
                episode_length_list.append(frame*(STEPS_TO_SKIP+1)/1000)
                if episode == 0:
                    avg_episode_length = frame*(STEPS_TO_SKIP+1)
                else:
                    avg_episode_length = (1-np.exp(-1/10))*frame*(STEPS_TO_SKIP+1) + np.exp(-1/10)*avg_episode_length
                avg_episode_length_list.append(avg_episode_length/1000)
                episode_score_list['player'].append(episode_score[0])
                episode_score_list['computer'].append(episode_score[1])
                u.plot_metrics(range(episode+1), 'EpisodeLength', 'Episode Length', 'Episode', 'Length/1000', (episode_length_list, 'Episode length'), (avg_episode_length_list, 'Average'))
                u.plot_metrics(range(episode+1), 'EpisodeScore', 'Episode Score', 'Episode', 'Score', (episode_score_list['player'], 'Player'), (episode_score_list['computer'], 'Computer'))
                
def play(G, save_path):
    
    # Initialize the Pong gym environment, set seeds
    env = gym.make('Pong-v0')
    replay_memory = u.ReplayMemory()
    with G.as_default():
        # Get input/output tensors
        X = G.get_tensor_by_name('X:0')
        Y = G.get_tensor_by_name('Y:0')        
        saver = tf.train.Saver(var_list=None, max_to_keep=5)
        # Initialize TF session
        with tf.Session() as sess:
            print('Reloading parameters...')
            saver.restore(sess, save_path)
            # Iterate over episodes
            while True:
                obs = u.preprocess_image(env.reset())
                for i in range(3):
                    replay_memory.add_frame(np.zeros((160//DOWNSAMPLE,160//DOWNSAMPLE)))
                replay_memory.add_frame(obs)
            
                # Iterate over frames
                done = False
                while not done:
                    # Feed state into DQN
                    s = np.stack([replay_memory.frames[i] for i in range(-4,0)], axis=-1).reshape(1,160//DOWNSAMPLE,160//DOWNSAMPLE,4)
                    y = sess.run(Y, feed_dict={X:s})
                    
                    # Decide on action
                    a = np.argmax(y)+1
                    
                    # Take action, observe environment, reward
                    obs, r, done, _ = env.step(a)
                    for i in range(STEPS_TO_SKIP):
                        obs, r, done_temp, _ = env.step(1)
                        if done_temp == True:
                            done = True
                    env.render()
                    
                    # Add new state/reward to replay memory
                    replay_memory.add_frame(u.preprocess_image(obs))
                
                q = input('play again?')
                if q in ['','y','Y']:
                    pass
                else:
                    env.render(close=True)
                    break




'''================================ TRAINING ======================================'''


#u.collect_pong_screens(max_episodes=1, steps_to_skip=STEPS_TO_SKIP, max_to_keep=128, ds_factor=DOWNSAMPLE)

if RELOAD_PARAMETERS:
    G = u.load_graph(SAVE_PATH)
else:
    G = build_Q_network(conv_layers=[((8,8,4,32),(1,4,4,1)), # 10x10 output
                                     ((4,4,32,64),(1,2,2,1)), # 3x3 output
                                     ((3,3,64,64),(1,1,1,1))], # 2x2 output
                        fc_layers=[(256,1), (256,1)], 
                        activation='relu')

if IS_TRAINING:
    train(G, MAX_EPISODES, SAVE_PATH)
else:
    play(G, SAVE_PATH)


































