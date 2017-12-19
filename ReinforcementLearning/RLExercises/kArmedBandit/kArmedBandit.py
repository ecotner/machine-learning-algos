#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:55:13 2017

A collection of k-armed bandit algorithms in order to practice reinforcement learning. This is simple action-value learning, as there is no "state" to speak of in these scenarios.

1) Epsilon-greedy algorithm

2) Nonstationary policy gradient

@author: ecotner
"""

import numpy as np
import matplotlib.pyplot as plt

class Bandit1(object):
    '''
    Class representing the epsilon-greedy k-armed bandit.
    Args:
        n_arms: number of arms on each bandit
        n_bandits: number of bandits to simulate at once
        epsilon: fraction of step in which the policy is random
        stochastic_reward: bool determining whether the rewards are a fixed value or a distribution
        alpha: step size parameter. If None, then step size is 1/N(a), where N(a) is the number of times action a has been chosen by a bandit.
    '''
    def __init__(self, n_arms=2, n_bandits=1, epsilon=0, stochastic_reward=False, alpha=None):
        self.n_arms = n_arms
        self.n_bandits = n_bandits
        self.epsilon = epsilon
        self.stochastic_reward = stochastic_reward
        self.alpha = alpha
        self.initialize()
    
    def initialize(self):
        ''' Initializes the possible rewards for each bandit for pulling each lever, the Q-values, and the action count. Creates a matrix called "mean_rewards", which is a matrix of reward averages. If stochastic, rewards are normally distributed with unit variance, centered on a mean uniformly distributed over (-5,5). '''
        self.mean_rewards = 2*2*(np.random.rand(self.n_bandits, self.n_arms) - 0.5)
        self.Q = 0.001*np.random.randn(self.n_bandits, self.n_arms)     # Action-value matrix Q
        self.n_actions = np.ones((self.n_bandits, self.n_arms))     # Counts the number of times each bandit pulls each lever
    
    def get_reward(self, actions):
        ''' Determines the rewards for each bandit for pulling a lever/arm, given a vector of actions of each of the bandits. '''
        if self.stochastic_reward == False:
            rewards = self.mean_rewards[np.arange(len(actions)), actions]
        else:
            rewards = np.random.randn(self.n_bandits) + self.mean_rewards[np.arange(len(actions)), actions]
        return rewards

    def pull_lever(self):
        ''' Each bandit chooses a lever to pull based on an epsilon-greedy policy, and the result is used to update the action value matrix Q '''
        # Determine whether action is greedy or random
        is_rand_step = np.random.rand(self.n_bandits) < self.epsilon
        # Take action, get rewards
        actions = is_rand_step*np.random.randint(self.n_arms, size=self.n_bandits) + (1-is_rand_step)*np.argmax(self.Q, axis=1)
        rewards = self.get_reward(actions)
        # Update Q matrix
        action_mask =  np.zeros((self.n_bandits, self.n_arms))
        action_mask[np.arange(self.n_bandits), actions] = 1     # Is one if given bandit takes given action
        self.n_actions += action_mask
        if self.alpha == None:
            self.Q += (rewards.reshape((self.n_bandits,1))*action_mask - self.Q)/self.n_actions
        else:
            self.Q += self.alpha*(rewards.reshape((self.n_bandits,1))*action_mask - self.Q)
        return action_mask

    def play_game(self, max_iterations):
        # Set up some stuff
        optimal_arm = np.argmax(self.mean_rewards, axis=1)    # Calculate optimal lever to pull for each bandit
        n_optimal = np.zeros((self.n_bandits))     # Number of times optimal lever is pulled
        frac_optimal_list = []
        # Iterate over lever pulls
        for step in range(max_iterations):
            # Pull the lever
            action_mask = self.pull_lever()
            # Compute some data
            is_optimal = (np.argmax(action_mask, axis=1) == optimal_arm)
            n_optimal += is_optimal
            frac_optimal = np.mean(is_optimal)
#            frac_optimal = np.mean(n_optimal/np.sum(self.n_actions, axis=1)) # Why isn't this right?
            frac_optimal_list.append(frac_optimal)
        return frac_optimal_list

def CompareBandits1(max_iterations, n_arms=3, n_bandits=100, epsilon=0, stochastic_reward=False, alpha=None):
    plt.figure('k-armed bandit')
    plt.clf()
    plt.title('Optimal pulls of {}-armed bandit'.format(n_arms))
    plt.xlabel('Iteration')
    plt.ylabel('Fraction')
    plt.ylim(0,1)
    if (type(epsilon) == list) or (type(stochastic_reward) == list):
        assert len(epsilon) == len(stochastic_reward), 'epsilon and stochastic_reward must be same length'
        for ep, sr in zip(epsilon, stochastic_reward):
            b = Bandit1(n_arms, n_bandits, ep, sr, alpha)
            x = b.play_game(max_iterations)
            plt.plot(x, label='$\epsilon$={}'.format(ep))
    else:
        b = Bandit1(n_arms, n_bandits, epsilon, stochastic_reward)
        x = b.play_game(max_iterations)
        plt.plot(x)
    plt.legend()

CompareBandits1(max_iterations=5000, n_arms=10, n_bandits=1000, epsilon=[0, 0.01, 0.1,], stochastic_reward=0*np.ones((3)))






















