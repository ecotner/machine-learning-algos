#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:12:59 2017

Example 4.3: Gambler’s Problem A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money. On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars. This problem can be formulated as an undiscounted, episodic, finite MDP. The state is the gambler’s capital, s ∈ {1, 2, . . . , 99} and the actions are stakes, a ∈ {0, 1, . . . , min(s, 100−s)}. The reward is zero on all transitions except those on which the gambler reaches his goal, when it is +1. The state-value function then gives the probability of winning from each state. A policy is a mapping
from levels of capital to stakes. The optimal policy maximizes the probability of reaching the goal. Let p_h denote the probability of the coin coming up heads. If p_h is known, then the entire problem is known and it can be solved, for instance, by value iteration.

Exercise 4.9: Implement value iteration for the gambler’s problem and solve it for p_h = 0.25 and p_h = 0.55. In programming, you may find it convenient to introduce two dummy states corresponding to termination with capital of 0 and 100, giving them values of 0 and 1 respectively.

@author: ecotner
"""

import numpy as np
import matplotlib.pyplot as plt

# Defining parameters of the scenario
n_states = 101 # Numbers 0 - 100
n_rewards = 2 # 0 and 1
n_actions = 51 # Can bet any number between 0 and 50

# Construct transition probability as a sparse array
def transition_probability(p_h):
    prob = np.zeros((n_states, n_rewards, n_states, n_actions))
    for s_ in range(n_states):
        for s in range(n_states):
            for a in range(min(s,100-s)+1):
                # Check if starting in terminal state
                if (s == 0) or (s == 100):
                    pass
                # Check if winning transition
                elif s_ == s + a:
                    # Check if terminal next state
                    if s_ == 100:
                        prob[s_,1,s,a] = p_h
                    else:
                        prob[s_,0,s,a] = p_h
                # Check if losing transiton
                elif s_ == s - a:
                    prob[s_,0,s,a] = 1-p_h
                # All other transitons get zero prob if they don't add up
    return prob

def value_iteration(p_h=0.5):
    # Initialize value function
    v0 = np.zeros((n_states))
    v1 = np.zeros((n_states))
    pi = np.zeros((n_states))
    prob = transition_probability(p_h)
    Delta = 1
    # Perform value iteration
    while Delta > 2e-3:
#    for i in range(50):
        Delta = 0
        for s in range(n_states):
            q = np.zeros((n_actions))
            for a in range(min(s,100-s)+1):
                for s_ in range(n_states):
                    for r in range(n_rewards):
                        q[a] += prob[s_,r,s,a] * (r + v0[s_])
            v1[s] = np.amax(q)
            Delta = np.max([Delta, np.max(np.abs(v0 - v1))])
            v0[s] = v1[s]
        print('Delta = {}'.format(Delta))
    # Extract policy from value function
    for s in range(n_states):
        q = np.zeros((n_actions))
        for a in range(min(s, 100-s)+1):
            for s_ in range(n_states):
                for r in range(n_rewards):
                    q[a] += prob[s_,r,s,a] * (r + v1[s_])
        pi[s] = np.argmax(q)
    return v1, pi

p_h = 0.4
v, pi = value_iteration(p_h)
plt.figure('policy')
plt.plot(pi)
plt.title('Optimal policy $\pi_*$, $p_h$={}'.format(p_h))
plt.xlabel('Captial')
plt.ylabel('Wager')
plt.figure('value')
plt.plot(v)
plt.title('Optimal value function $v$, $p_h$={}'.format(p_h))
plt.xlabel('Capital')
plt.ylabel('Value')

























