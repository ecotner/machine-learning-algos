#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:29:52 2017

Problem: Jack runs two car rental locations, and must determine how to distribute his rental cars between the two locations. He earns $10 for each car rental, and they become available for rental the day after they are returned. It costs him $2 to move cars between each location. The number of cars requested at each location is a Poisson random variable n. To simplify the problem, we assume that each location can hold a maximum number of cars, and similarly, only a maximum number can be transferred between each location on a given day.

@author: ecotner
"""

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

def poisson(n, mean):
    return math.exp(-mean)*(mean**n)/math.factorial(n)

def clipped_poisson(n, mean, maximum=10):
    ''' Since the number of cars can't go above a certain limit, we move all the probability mass above the maximum into the probability of the maximum. '''
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
        self.n_locations = 2
        self.max_cars = 10
        self.max_transfer = 5
        self.mean_rented = [3,4]
        self.mean_returned = [3,2]
        self.reward_rented = 10
        self.reward_trans = -2
        assert self.n_locations == len(self.mean_rented) == len(self.mean_returned), "Number of locations inconsistent"
        self.states = [(i,j) for i in range(self.max_cars) for j in range(self.max_cars)]
        self.value = np.zeros([self.max_cars for i in range(self.n_locations)])
    
    def transition_prob(self):
        ''' Calculates the transition probability to s_ with reward r given initial state s and action a. Creates an array which is saved as a .npy file which can be reloaded later. 
        ''' 
#        prob = np.zeros((self.max_cars, self.max_cars, self.reward_rented*self.n_locations*self.max_cars - self.reward_trans*self.max_transfer, self.max_cars, self.max_cars, 1+2*self.max_transfer))
        prob = {}
#        for n1 in range(self.max_cars+1):
#            for n2 in range(self.max_cars+1):
#                print("n1={}, n2={}".format(n1, n2))
#                for n1_rented in range(n1+1):
#                    for n2_rented in range(n2+1):
#                        for n1_returned in range((self.max_cars - max(0,n1-n1_rented))+1):
#                            for n2_returned in range((self.max_cars - max(0,n2-n2_rented))+1):
#                                for a in range(-self.max_transfer, self.max_transfer+1):
        for n1 in range(self.max_cars+1):
            for n2 in range(self.max_cars+1):
                print("n1={}, n2={}".format(n1, n2))
                for a in range(max(-self.max_transfer,-n1,n2-self.max_cars), min(self.max_cars-n1,n2,self.max_transfer)+1):
                    for n1_rented in range(min(self.max_cars,n1+a)+1):
                        for n2_rented in range(min(self.max_cars,n2-a)+1):
                            for n1_returned in range(self.max_cars-(n1+a-n1_rented)+1):
                                for n2_returned in range(self.max_cars-(n2-a-n2_rented)+1):
                                    n1_ = min(self.max_cars, max(0, n1+a-n1_rented)+n1_returned)
                                    n2_ = min(self.max_cars, max(0, n2-a-n2_rented)+n2_returned)
                                    r = self.reward_rented*(min(n1+a,n1_rented) + min(n2-a,n2_rented)) + self.reward_trans*abs(a)
                                    prob_ = clipped_poisson(n1_rented, self.mean_rented[0],min(self.max_cars,n1+a))*clipped_poisson(n2_rented, self.mean_rented[1],min(self.max_cars,n2-a))*clipped_poisson(n1_returned, self.mean_returned[0],self.max_cars-(n1+a-n1_rented))*clipped_poisson(n2_returned, self.mean_returned[1],self.max_cars-(n2+a-n2_rented))
                                    prob[(n1_, n2_, r, n1, n2, a)] = prob.get((n1_, n2_, r, n1, n2, a), 0) + prob_
        return prob


env = Environment()
prob = env.transition_prob()


#plt.figure('test')
#plt.plot([clipped_poisson(n, mean=5, maximum=10) for n in range(15)])





























