#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:29:52 2017

Problem: Jack runs two car rental locations, and must determine how to distribute his rental cars between the two locations. He earns $10 for each car rental, and they become available for rental the day after they are returned. It costs him $2 to move cars between each location. The number of cars requested at each location is a Poisson random variable n. To simplify the problem, we assume that each location can hold a maximum number of cars, and similarly, only a maximum number can be transferred between each location on a given day.

We can then add in some additional reward structure. One of Jack’s employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle one car to the second location for free. Each additional car still costs $2, as do all cars moved in the other direction. In addition, Jack has limited parking space at each location. If more than 5 cars are kept overnight at a location (after any moving of cars), then an additional cost of $4 must be incurred to use a second parking lot (independent of how many cars are kept there).

@author: ecotner
"""

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

gamma = 0.9

def poisson(n, mean):
    return math.exp(-mean)*(mean**n)/math.factorial(n)

def clipped_poisson(n, mean, maximum=10):
    ''' Since the number of cars rented or returned can't go above a certain limit (due to max cars allowed on lot), we move all the probability mass above the maximum into the probability of the maximum. '''
    if n == maximum:
        return 1 - sum([poisson(m, mean) for m in range(maximum)])
    elif (n < 0) or (n > maximum):
        return 0
    else:
        return poisson(n, mean)

# Create environment class for agent to interact with
class Environment(object):
    '''
    Class representing the environment; in this case, the car rental business. The states in which the environment can exist consist of an array numbers representing the number of cars at each location. The transistions between states are the result of 1) the number of cars the agent decides to transfer between locations 2) the number of cars rented/returned that day.
    '''
    def __init__(self):
        self.is_original = False
        self.n_locations = 2
        self.max_cars = 10
        self.max_transfer = 5
        self.mean_rented = [3,4]
        self.mean_returned = [3,2]
        self.reward_rented = 10
        self.reward_trans = -2
        self.reward_overnight_parking = -4
        assert self.n_locations == len(self.mean_rented) == len(self.mean_returned), "Number of locations inconsistent"
        self.states = [(i,j) for i in range(self.max_cars) for j in range(self.max_cars)]
        self.value = np.zeros([self.max_cars for i in range(self.n_locations)])
    
    def transition_prob(self):
        ''' Calculates the transition probability to s_ with reward r given initial state s and action a. Creates a dictionary which contains the probability densities for the specified inputs. 
        ''' 
        prob = {}
        # Iterate over all possible starting values of cars
        for n1 in range(self.max_cars+1):
            for n2 in range(self.max_cars+1):
                print("n1={}, n2={}".format(n1, n2))
                # Iterate over all possible transfers between lots
                for a in range(max(-self.max_transfer,-n1,n2-self.max_cars), min(self.max_cars-n1,n2,self.max_transfer)+1):
                    # Iterate over all possible combinations of rentals and returns
                    for n1_rented in range(min(self.max_cars,n1+a)+1):
                        for n2_rented in range(min(self.max_cars,n2-a)+1):
                            for n1_returned in range(self.max_cars-(n1+a-n1_rented)+1):
                                for n2_returned in range(self.max_cars-(n2-a-n2_rented)+1):
                                    # Calculate number of cars in each lot after state transition, and reward from new state
                                    n1_ = min(self.max_cars, max(0, n1+a-n1_rented)+n1_returned)
                                    n2_ = min(self.max_cars, max(0, n2-a-n2_rented)+n2_returned)
                                    if self.is_original:
                                        r = self.reward_rented*(min(n1+a,n1_rented) + min(n2-a,n2_rented)) + self.reward_trans*abs(a)
                                    else:
                                        if a > 0:
                                            r = self.reward_rented*(min(n1+a,n1_rented) + min(n2-a,n2_rented)) + self.reward_trans*a
                                        else:
                                            r = self.reward_rented*(min(n1+a,n1_rented) + min(n2-a,n2_rented)) + self.reward_trans*(a+1)
                                        if n1_ > self.max_cars//2:
                                            r += self.reward_overnight_parking
                                        if n2_ > self.max_cars//2:
                                            r += self.reward_overnight_parking
                                    # Calculate probability of transition occuring this way
                                    prob_ = clipped_poisson(n1_rented, self.mean_rented[0],min(self.max_cars,n1+a))*clipped_poisson(n2_rented, self.mean_rented[1],min(self.max_cars,n2-a))*clipped_poisson(n1_returned, self.mean_returned[0],self.max_cars-(n1+a-n1_rented))*clipped_poisson(n2_returned, self.mean_returned[1],self.max_cars-(n2-a-n2_rented))
                                    # Since new states are combinations of different ways the transition can happen, we need to sum the probabilities of all the different ways a given initial state can end up in the same final state
                                    prob[(n1_, n2_, r, n1, n2, a)] = prob.get((n1_, n2_, r, n1, n2, a), 0) + prob_
        return prob

# Create environment and calculate all possible transition probabilities (uncomment to evaluate before doing anything else)
env = Environment()
prob = env.transition_prob()

# Unit tests to ensure the transition probability is accurate
def check_normalization(n1, n2, a):
    ''' Checks that transition probability is properly normalized for given n1, n2, and a. '''
    p = 0
    for n1_ in range(env.max_cars+1):
        for n2_ in range(env.max_cars+1):
            for r in range(2*env.reward_overnight_parking + env.reward_trans * env.n_locations * env.max_transfer, env.reward_rented * env.n_locations * env.max_cars+1):
                p += prob.get((n1_,n2_,r,n1,n2,a), 0)
    return p

def check_meta_normalization(delta=0.01):
    ''' Checks that normalization of transition probability is close to unity for all possible combinations of conditioned variables n1, n2, and a. '''
    tot = 0
    n = 0
    for n1 in range(env.max_cars+1):
        for n2 in range(env.max_cars+1):
            print("n1={}, n2={}".format(n1, n2))
            for a in range(max(-env.max_transfer,-n1,n2-env.max_cars), min(env.max_cars-n1,n2,env.max_transfer)+1):
                tot += 1
                norm = check_normalization(n1,n2,a)
                if (abs(1-norm) < delta) or (abs(norm) < delta):
                    n += 1
    return n/tot

def policy_evaluation(v1, policy):
    Delta = 1
    while Delta > .1:
        Delta = 0
        v2 = {(n1,n2):0 for n1 in range(env.max_cars+1) for n2 in range(env.max_cars+1)}
        v0 = v1
        for n1 in range(env.max_cars+1):
            for n2 in range(env.max_cars+1):
                for r in range(env.n_locations * env.reward_overnight_parking + env.reward_trans * env.n_locations * env.max_transfer, env.reward_rented * env.n_locations * env.max_cars+1):
                    for n1_ in range(env.max_cars+1):
                        for n2_ in range(env.max_cars+1):
                            v2[(n1,n2)] += prob.get((n1_,n2_,r,n1,n2,policy[(n1,n2)]), 0) * (r + gamma*v1[(n1_,n2_)])
                v1[(n1,n2)] = v2[(n1,n2)]
                Delta = max(Delta, abs(v1[(n1,n2)]-v0[(n1,n2)]))
    return v1

def policy_improvement(v, policy):
    policy_stable = True
    for n1 in range(env.max_cars+1):
        for n2 in range(env.max_cars+1):
            pi0 = policy[(n1,n2)]
            Q = []
            for a in range(-env.max_transfer,env.max_transfer+1):
                q = 0
                for n1_ in range(env.max_cars+1):
                    for n2_ in range(env.max_cars+1):
                        for r in range(env.n_locations * env.reward_overnight_parking + env.reward_trans * env.n_locations * env.max_transfer, env.reward_rented * env.n_locations * env.max_cars+1):
                            q += prob.get((n1,n2,r,n1_,n2_,a),0) * (r + gamma*v[(n1_,n2_)])
                Q.append(q)
            policy[(n1,n2)] = np.argmax(Q)-5 # Offset since index 0 of Q corresponds to action of -5
            if pi0 != policy[(n1,n2)]:
                policy_stable = False
    return policy, policy_stable

def policy_iteration():
    # Initialize "blank" policy and value function
    pi0 = {(n1,n2):0 for n1 in range(env.max_cars+1) for n2 in range(env.max_cars+1)}
    v0 = {(n1,n2):10 for n1 in range(env.max_cars+1) for n2 in range(env.max_cars+1)}
    policy_stable = False
    i=0
    # Iterate policy
    while policy_stable == False:
        i += 1
        # Evaluate current policy
        v1 = policy_evaluation(v0, pi0)
        contour_plot(v1, 'Value function i={}'.format(i), 'viridis')
        # Improve current policy and check for stability
        pi1, policy_stable = policy_improvement(v1, pi0)
        contour_plot(pi1, 'Policy i={}'.format(i), 'plasma')
        v0 = v1
        pi0 = pi1
    return v0, pi0

def contour_plot(v, name='Value function', colormap='viridis'):
    # Convert dictionary to array
    v_array = np.zeros((env.max_cars+1,env.max_cars+1))
    for n1 in range(env.max_cars+1):
        for n2 in range(env.max_cars+1):
            v_array[n1,n2] = v[(n1,n2)]
    plt.figure(name)
    c = plt.contourf(v_array, cmap=colormap)
    plt.title(name)
    plt.xlabel('# cars, lot 1')
    plt.ylabel('# cars, lot 2')
    plt.colorbar(c)
    plt.show()

# Compute the optimal value function and policy
v, pi = policy_iteration()

















