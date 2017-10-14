# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 21:00:26 2017

Exercise in Q-learning.

Object is to traverse a grid which is composed of 'frozen' patches (1), 
holes (0), a starting point (2), and a goal (3). Moving into the edges of the 
lake does nothing.

The problem is handled by using a Q-value in order to determine the most effective
action based on the current situation

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

''' ==================== CREATE FROZEN LAKE ENVIRONMENT =================== '''

class Lake(object):
    ''' Class which represents the 'frozen lake' maze. '''
    hole_penalty = -1
    
    def __init__(self, size=4):
        self.grid = self.create_grid(size)
        self.size = size
        self.loc = [0,0]
        self.next_loc = [0,0]
        self.state = 0
        self.next_state = None
    
    def create_grid(self, size):
        # Initialize blank grid
        grid = np.empty((size,size), dtype=int)
        # Place start and goal in opposite corners of grid
        grid[0,0] = 4
        grid[size-1,size-1] = 3
        # Generate random path between start and goal
        coords = [0,0]
        while True:
            if coords[0] == size-1:
                coords[1] += 1
            elif coords[1] == size-1:
                coords[0] += 1
            else:
                if np.random.rand() < 0.5:
                    coords[0] += 1
                else:
                    coords[1] += 1
            if grid[coords[0],coords[1]] == 3:
                break
            else:
                grid[coords[0],coords[1]] = 1
#            print(grid)
        for i in range(size):
            for j in range(size):
                if grid[i,j] not in (4,1,3):
                    if np.random.rand() < 0.3:
                        grid[i,j] = 1
                    else:
                        grid[i,j] = 0
        grid[0,0] = 2
        return grid
    
    def get_state(self):
        return self.loc[0] + self.size*self.loc[1]
    
    def get_next_state(self):
        return self.next_loc[0] + self.size*self.next_loc[1]
    
    def get_reward(self):
        if self.grid[self.loc[0],self.loc[1]] == 3:
            return 1
        elif self.grid[self.loc[0],self.loc[1]] == 0:
            return self.hole_penalty
        else:
            return (0.01/(1.41*self.size))*np.sqrt(self.loc[0]**2 + self.loc[1]**2)
    
    def get_next_reward(self):
        if self.grid[self.next_loc[0],self.next_loc[1]] == 3:
            return 1
        elif self.grid[self.next_loc[0],self.next_loc[1]] == 0:
            return self.hole_penalty
#        elif self.next_state == -1:
#            return -1
        else:
            return 0
    
    def take_step(self, action):
        ''' 
        Updates the state based on input action. Doesn't actually update the
        step, so you can use this to explore the adjacent area to the current
        location.
        Inputs: 0=right, 1=up, 2=left, 3=down
        '''
        if action == 0:
            if self.loc[1] == self.size-1:
#                self.next_state = -1
                pass
            else:
                self.next_loc[1] = self.loc[1] + 1
        elif action == 1:
            if self.loc[0] == 0:
#                self.next_state = -1
                pass
            else:
                self.next_loc[0] = self.loc[0] - 1
        elif action == 2:
            if self.loc[1] == 0:
#                self.next_state = -1
                pass
            else:
                self.next_loc[1] = self.loc[1] - 1
        elif action == 3:
            if self.loc[0] == self.size-1:
#                self.next_state = -1
                pass
            else:
                self.next_loc[0] = self.loc[0] + 1
        self.next_state = self.get_next_state()
    
    def update_state(self):
        if self.next_state == -1:
            pass
        else:
            self.state = self.get_next_state()
            self.loc = self.next_loc[:]
    
    def reset_state(self):
        self.loc = [0,0]
        self.state = self.get_state()
        self.next_loc = [0,0]
        self.next_state = None
    
    def is_done(self):
        state_flag = self.grid[self.loc[0],self.loc[1]]
        if state_flag == 0 or state_flag == 3:
            return True
        else:
            return False
#        if (self.state == -1 or self.state == self.size**2 - 1 or
#            self.grid[self.loc[0],self.loc[1]] == 0):
#            return True
#        else:
#            return False
    
    def __str__(self):
        return str(self.grid)

def one_hot(n,m):
    ''' Returns a one-hot vector of shape (1,m) with a one in position n 
    (zero-indexed) '''
    vec = np.zeros((1,m))
    vec[0,n] = 1
    return vec

''' ====================== SET UP THE NETWORK ============================ '''

# Create the 'frozen lake' and display
lake_size = 8
lake = Lake(lake_size)
print(lake)

# Set up computation graph in TF
tf.reset_default_graph()
xavier_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

# Simple linear network
#input_layer = tf.placeholder(tf.float32, shape=[1,lake_size**2], name='input_layer')
#W1 = tf.get_variable('W1', shape=[lake_size**2,4], initializer=xavier_init)
#Q_out = tf.matmul(input_layer, W1, name='Q_out')

# Neural network
input_layer = tf.placeholder(tf.float32, shape=[1,lake_size**2], name='input_layer')
W1 = tf.get_variable('W1', shape=[lake_size**2,100], initializer=xavier_init)
b1 = tf.Variable(np.zeros((1,100)), dtype=tf.float32, name='b1')
A1 = tf.nn.relu(tf.matmul(input_layer, W1) + b1, name='A1')
W2 = tf.get_variable('W2', shape=[100,4], initializer=xavier_init)
Q_out = tf.matmul(A1, W2, name='Q_out')

action_pred = tf.argmax(Q_out, axis=1, name='action_pred')
Q_target = tf.placeholder(tf.float32, shape=[1,4], name='Q') # This will be target Q
loss = 0.5*tf.reduce_sum((Q_out - Q_target)**2)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

''' ====================== TRAIN THE NETWORK =========================== '''

num_episodes = 2000
discount = 0.99
e0 = 0.1

with tf.Session() as sess:
    # Initialize stuff
    init = tf.global_variables_initializer()
    sess.run(init)
    total_reward = 0
    train_loss_list = []
    reward_list = []
    j_list = []
    e = float(e0)
    # Iterate over multiple episodes
    for i in range(num_episodes):
        # Start walker from the beginning
        lake.reset_state()
        done = False
        # Let the walker loose!
        for j in range(500):
            # Get predicted action and Q-values
            state = lake.get_state()
            a, Q_pred = sess.run([action_pred, Q_out], 
                feed_dict={input_layer:one_hot(state,lake_size**2)})
            if np.random.rand() < e:
                a[0] = np.random.choice(4)
            # Explore in the direction of the prediction
            lake.take_step(a[0])
            lake.update_state()
            next_state = lake.get_state()
            next_reward = lake.get_reward()
            if next_reward == 1:
                total_reward += next_reward
            done = lake.is_done()
            # Get prediction of next Q-values
            Q_next = sess.run(Q_out, feed_dict={input_layer:one_hot(next_state,lake_size**2)})
            # Train the network
            Q = Q_pred.copy()
            Q[0,a[0]] = next_reward + discount*np.max(Q_next)
            _, train_loss = sess.run([train_op, loss], feed_dict={input_layer:one_hot(state,lake_size**2), Q_target:Q})
            # Update the state
            
            # End episode if walker dies or finishes
            if done == True:
                print('Episode: {}, total reward: {}, loss: {}'.format(i, total_reward, train_loss))
                train_loss_list.append(train_loss)
                reward_list.append(total_reward)
                j_list.append(j+1)
                e = e0/((total_reward/1000)**2 + 1)
                break
    W_matrix = sess.run(W1)

print('Fraction of successful episodes: {}'.format(total_reward/num_episodes))

plt.figure('training loss')
plt.clf()
plt.semilogy(train_loss_list)
plt.title('Training loss at end of episode')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.figure('reward')
plt.clf()
plt.plot(reward_list)
plt.title('Cumulative reward count')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.figure('steps through level')
plt.clf()
plt.semilogy(j_list)
plt.title('Steps through level')
plt.xlabel('Episode')
plt.ylabel('Steps')
print(lake)





























