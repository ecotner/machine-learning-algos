# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:30:15 2017

@author: 27182_000
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Data point generation function
def dataGen(k, n, x, y):
    '''
    Data point generation function. Initializes k clusters (with n//k points each) drawn from a Gaussian distribution,
    whose means are uniformly distributed over the space S=[0,x]*[0,y] with unit variance. Points which fall
    outside of S are discarded and another point is drawn.
    '''
    # Define recursive function for picking new data points if outside S
    def pickPoints(mu, k, data_points):
        # points will contain (x, y, index of mean)
        point = (random.gauss(mu[0],1), random.gauss(mu[1],1), k)
        if (0 <= point[0] <= x) and (0 <= point[1] <= y):
            data_points.append(point)
        else:
            pickPoints(mu, k, data_points)
    # Initialize cluster means
    Mu = np.array([(random.uniform(0,x), random.uniform(0,y)) for i in range(k)])
    # Draw points from Gaussian distributions
    data_points = []
    for i in range(n//k):
        for j in range(k):
            pickPoints(Mu[j], j, data_points)
    # Return tuple which is (vector of data points, vector of means
    return np.array(data_points)

# unit testing above function
#(k,n,x,y) = (3,100,10,10)
#data = dataGen(k, n, x, y)
#data_k = np.zeros(k).tolist()
#for i in range(k):
#    data_k[i] = data[data[:,2] == i]
#    plt.scatter(data_k[i][:,0], data_k[i][:,1])

# k-means clustering algorithm
def kMeans(k, n, x, y):
    '''
    Algorithm for finding clusters of data in a 2D plane of size xy. First, n data points are drawn from k Gaussians
    with uniformly distributed means and unit variance. Then, the data is partitioned into k clusters using
    the k-means clustering algorithm.
    Args:
        k: number of clusters
        n: number of data points
        x,y: dimensions of 2D plane
    '''
    # Define function for calculating Euclidean distance of data point from centroid
    def euclideanDist(x, mu):
        return np.linalg.norm(x-mu)
    # Initialization - clear plots, generate data points, randomize centroids
    plt.figure(1)
    plt.clf()
    data = dataGen(k, n, x, y)
    X = data[:,:2]
    m = X.shape[0]
    y = data[:,2]
    Mu = X[np.random.choice(range(m), size=k, replace=False)][:,:2]
    # Create flag for determining loop condition
    flag = 0
    # Plot initial positions
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.plot(X[:,0], X[:,1], c='k', marker='o', linestyle='None')
    plt.show()
    for i in range(k):
        plt.plot(Mu[i,0], Mu[i,1], c=colors[i], marker='x', markersize=20, markeredgewidth=5, linestyle='None')
    plt.show()
    plt.pause(1)
    # Loop until convergence
    while flag == 0:
        # Determine data-centroid associations
        # Cluster association will be held in a list of length k, where each element is itself a list
        #   containing the indices of X which belong to each centroid
        clusters = [[] for i in range(k)]
        # Loop over all data point indices
        for i in range(m):
            closest_dist = 10**5
            # Loop over all centroid indices
            for j in range(k):
                # Calculate distance between data point and centroid, save closest centroid index
                dist = euclideanDist(X[i], Mu[j])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = j
            clusters[closest_idx].append(i)
        # Plot updated centroid positions and data association?
        plt.clf()
        for i in range(k):
            plt.plot(X[clusters[i]][:,0], X[clusters[i]][:,1], c=colors[i], marker='o', linestyle='None')
            plt.plot(Mu[i,0], Mu[i,1], c=colors[i], marker='x', markersize=20, markeredgewidth=5, linestyle='None')
        plt.show()
        plt.pause(.5)
        # Calculate/update new centroid positions, check for convergence
        Mu_old = Mu.copy()
        # Loop over centroid indices
        for i in range(k):
            # Calculate mean (x,y) of each cluster, set to new mu
            Mu[i,:] = np.mean(X[clusters[i]], axis=0)
        # Check for convergence
        if np.all(Mu == Mu_old):
            flag = 1

# Test kMeans repeatedly, display interactive plot in another window
plt.ion()
while True:
    kMeans(4,100,10,10)















