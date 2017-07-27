# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:19:35 2017

Simple k-nearest neighbors algorithm for predicting association.

@author: 27182_000
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# a dot-product metric; calculates X.Y/|X||Y| = cos(angle between X and Y)
def dotProduct(X, Y):
    assert len(X) == len(Y), 'vectors not the same length'
    dist = 0
    magX2, magY2 = 0, 0
    for i in range(len(X)):
        dist += X[i]*Y[i]
        magX2 += X[i]**2
        magY2 += Y[i]**2
    return -dist/math.sqrt(magX2*magY2)

# a Euclidean metric
def euclideanDist(X, Y):
    assert len(X) == len(Y), 'vectors not the same length'
    dist2 = 0
    for i in range(len(X)):
        dist2 += (X[i] - Y[i])**2
    return math.sqrt(dist2)

# algorithm for finding k smallest elements in a list
def min_k_indices(L, k):
    ''' finds the indices of the k smallest values in the list L '''
    assert len(L) >= k
    # initialize list of indices
    min_indices = [0]
    # iterate over the index of each element in L
    for i in range(1,len(L)):
        # iterate over indices in min_indices
        for j in range(len(min_indices)):
            # if the element of L is smaller than the element associated with the index, set the index to this one
            if L[i] < L[min_indices[j]]:
                min_indices.insert(j, i)
                if len(min_indices) > k:
                    del min_indices[-1]
                break
            if len(min_indices) < k and len(min_indices) == j+1:
                min_indices.append(i)
    return min_indices

# for testing the above function
#random.seed(1)
#vec = [random.random() for i in range(10)]
#min_indices = min_k_indices(vec,10)
#print(vec)
#print(min_indices)
#sorted_vec = []
#for e in min_indices:
#    sorted_vec.append(vec[e])
#print(sorted_vec)

# algorithm for getting the most frequently occurring element in a list
def getMostFrequent(L):
    assert len(L) > 0, 'Can\'t get most frequent element of empty list'
    # initialize dictionary to hold frequency data, have default key with value zero for comparison purposes
    maxKey = 'DEFAULT_KEY'
    freq_dict = {maxKey:0}
    # iterate over each element in L
    for e in L:
        # if element is already in dictionary, increase count
        if e in freq_dict:
            freq_dict[e] += 1
        # else add it to the dictionary and set count to one
        else:
            freq_dict[e] = 1
    # iterate over each key in the frequency dictionary
    for key in freq_dict:
        # if value of key is larger than the current largest value, make it the new max key
        if freq_dict[key] > freq_dict[maxKey]:
            maxKey = key
    # return max key
    return maxKey

#print(getMostFrequent([1,2,3,None,5,6,None,8]))

# define kNN algorithm, might need pre-processing of data beforehand so that it all scales nicely
def kNNregression(training_feature_vectors, classification_feature_vectors, feature_indices_to_predict=[-1], k_neighbors=1, metric=None):
    '''
    k-nearest neighbors algorithm to classify data (actually regression). Calculates the distance between elements in
    the feature space, finds the k nearest neighbors, and fills in the missing features with the mean of the features
    of the k nearest neighbors.
    Args:
        training_feature_vectors: list of feature vectors (also lists), containing classification of training data
        classification_feature_vectors: list of feature vectors (minus the features to be predicted) which we want
            to be classified
        feature_indices_to_predict: list of integers corresponding to indices in training_feature_vectors which are
            to be predicted for classification_feature_vectors, sorted in increasing order
        k_neighbors: integer number of neighbors to compare to
        metric: function of two feature vectors which determines distance from each other
        scaling: scales the elements of the feature vector
    Returns:
        classified_feature_vectors: the completed list of classification_feature_vectors with their missing features
            classified.
    '''
    # assertions on inputs
    if type(feature_indices_to_predict) == int:
        feature_indices_to_predict = [feature_indices_to_predict]
    elif type(feature_indices_to_predict) not in [list, type(np.array([]))]:
        assert False, 'feature_indices_to_predict is neither list nor integer type.'
    # split training_feature_vectors into two arrays - one to hold elements of feature vector that will be used to
    #   compute distance between objects, and another to hold elements of feature vector that will be used for classification
    #   these will be called 'reduced' feature vectors
    red_train_feat_vecs, red_class_feat_vecs = [], []
    for tvec in training_feature_vectors:
        red_tvec = []
        red_train_feat_vecs.append(red_tvec)
        for j in range(len(tvec)):
            if j not in feature_indices_to_predict:
                red_tvec.append(tvec[j])
    for cvec in classification_feature_vectors:
        red_cvec = []
        red_class_feat_vecs.append(red_cvec)
        for j in range(len(cvec)):
            if j not in feature_indices_to_predict:
                red_cvec.append(cvec[j])
    # compute distance between reduced training vectors and classification vectors, store in array where the 0 axis is
    #   the index of the vector to be classified, and the 1 axis is the distance between it and all training vectors
    distances = []
    for red_cvec in red_class_feat_vecs:
        dist_vec = []
        distances.append(dist_vec)
        for red_tvec in red_train_feat_vecs:
            dist_vec.append(metric(red_cvec, red_tvec))
    # now we need to find the indices of the k smallest distances for each list in distances (lists of indices of k
    #   nearest neighbors)
    min_dist_indices = []
    for cdist in distances:
        min_dist_indices.append(min_k_indices(cdist, k_neighbors))
    # for each vector to be classified, take average of the features of k nearest neighbors that are missing from the
    #   classification data
    classification_feature_vectors_ = copy.copy(classification_feature_vectors)
    tvecs_array = np.array(training_feature_vectors)
    for i in range(len(classification_feature_vectors)):
        for j in feature_indices_to_predict:
            feat_mean = 0
            for k in min_dist_indices[i]:
                feat_mean += tvecs_array[k,j]
            feat_mean /= k_neighbors
            classification_feature_vectors_[i][j] = feat_mean
    return classification_feature_vectors_

# some unit tests to make sure the regression algorithm works
def nDimRegressionUnitTest(num_data_points, ndim, missing_comps):
    '''
    testing it out: generates a bunch of random n-dimensional vectors, then tries to guess the missing components of
    a vector to be classified by comparing to a number of nearest neighbors using a specified metric.
    '''
    random.seed(0)
    tvecs = [[random.randint(0,9) for i in range(n_dim)] for j in range(num_data_points)]
    cvecs = [[random.randint(0,9) if i not in missing_comps else None for i in range(n_dim)] for j in range(1)]
    print('k=0: {}'.format(cvecs))
    for k in range(1,25,2):
        print('k={}: {}'.format(k, kNNregression(tvecs, cvecs, missing_comps, k, metric=dotProduct)))

#nDimRegressionUnitTest(100, 5, [1,4])

def twoDimRegressionUnitTest(num_data_points, num_class_points, k=3, class_dist='gauss', dist=3, sigma=1):
    '''
    Generates a bunch of Gaussian-distributed (x,y) coordinates as training data, then attempts to guess the y value of
    a classification point given its x value. Plots the results.
    '''
#    random.seed(0)
    mean_xy = dist/(2*math.sqrt(2))
    tvecs = [[random.gauss(-mean_xy, sigma) if j <= num_data_points//2 else random.gauss(mean_xy, sigma) for i in range(2)] for j in range(num_data_points)]
    if class_dist == 'gauss':
        cvecs = [[random.gauss(-mean_xy, sigma) if j <= num_class_points//2 else random.gauss(mean_xy, sigma), None] for j in range(num_class_points)]
    elif class_dist == 'uniform':
        cvecs = [[random.uniform(-(mean_xy+2*sigma), mean_xy+2*sigma), None] for i in range(num_class_points)]
    else:
        assert False, 'Choose either \'gauss\' or \'uniform\' distribution'
    cvecs_fit = kNNregression(tvecs, cvecs, [1], k, metric=euclideanDist)
    plt.figure('scatter')
    tvecs_array, cvecs_fit_array = np.array(tvecs), np.array(cvecs_fit)
    plt.scatter(tvecs_array[:,0], tvecs_array[:,1], c='k', label='Training data')
    plt.scatter(cvecs_fit_array[:,0], cvecs_fit_array[:,1], c='r', label='Fit data')
    plt.title('kNN fit to {} training points with k={}'.format(num_data_points, k))
    plt.show()

# Try test with k small and k large and compare - the large k tends to guess a y value near the mean of the distribution
#   (as it should), with very little variablility, whereas the small k tends to trace the training data distribution
#   more faithfully.
#twoDimRegressionUnitTest(1000,100, k=3, class_dist='gauss', dist=5, sigma=0.5)


# define kNN algorithm, might need pre-processing of data beforehand so that it all scales nicely
def kNNclassification(training_feature_vectors, classification_feature_vectors, feature_indices_to_predict=[-1], k_neighbors=1, metric=None):
    '''
    k-nearest neighbors algorithm to classify data. Calculates the distance between elements in
    the feature space, finds the k nearest neighbors, and fills in the missing features with the majority of the features
    of the k nearest neighbors.
    Args:
        training_feature_vectors: list of feature vectors (also lists), containing classification of training data
        classification_feature_vectors: list of feature vectors (minus the features to be predicted) which we want
            to be classified
        feature_indices_to_predict: list of integers corresponding to indices in training_feature_vectors which are
            to be predicted for classification_feature_vectors, sorted in increasing order
        k_neighbors: integer number of neighbors to compare to
        metric: function of two feature vectors which determines distance from each other
        scaling: scales the elements of the feature vector
    Returns:
        classified_feature_vectors: the completed list of classification_feature_vectors with their missing features
            classified.
    '''
    # assertions on inputs
    if type(feature_indices_to_predict) == int:
        feature_indices_to_predict = [feature_indices_to_predict]
    elif type(feature_indices_to_predict) not in [list, type(np.array([]))]:
        assert False, 'feature_indices_to_predict is neither list nor integer type.'
    # split training_feature_vectors into two arrays - one to hold elements of feature vector that will be used to
    #   compute distance between objects, and another to hold elements of feature vector that will be used for classification
    #   these will be called 'reduced' feature vectors
    red_train_feat_vecs, red_class_feat_vecs = [], []
    for tvec in training_feature_vectors:
        red_tvec = []
        red_train_feat_vecs.append(red_tvec)
        for j in range(len(tvec)):
            if j not in feature_indices_to_predict:
                red_tvec.append(tvec[j])
    for cvec in classification_feature_vectors:
        red_cvec = []
        red_class_feat_vecs.append(red_cvec)
        for j in range(len(cvec)):
            if j not in feature_indices_to_predict:
                red_cvec.append(cvec[j])
    # compute distance between reduced training vectors and classification vectors, store in array where the 0 axis is
    #   the index of the vector to be classified, and the 1 axis is the distance between it and all training vectors
    distances = []
    for red_cvec in red_class_feat_vecs:
        dist_vec = []
        distances.append(dist_vec)
        for red_tvec in red_train_feat_vecs:
            dist_vec.append(metric(red_cvec, red_tvec))
    # now we need to find the indices of the k smallest distances for each list in distances (lists of indices of k
    #   nearest neighbors)
    min_dist_indices = []
    for cdist in distances:
        min_dist_indices.append(min_k_indices(cdist, k_neighbors))
    # for each vector to be classified, take majority of the features of k nearest neighbors that are missing from the
    #   classification data
    classification_feature_vectors_ = copy.deepcopy(classification_feature_vectors)
    for i in range(len(classification_feature_vectors)):
        for j in feature_indices_to_predict:
                majority_classification = getMostFrequent([training_feature_vectors[k][j] for k in min_dist_indices[i]])
                classification_feature_vectors_[i][j] = majority_classification
    return classification_feature_vectors_

# some unit tests to make sure the classification algorithm works
def twoClassificationUnitTest(num_data_points, num_class_points, k=3, dist=3, sigma=1):
    '''
    Generates a two bunches of Gaussian-distributed (x,y) coordinates as training data (each labeled as red or blue), then
    attempts to classify a bunch of randomly distributed points as red/blue. Plots the results.
    '''
    random.seed(0)
    mean_xy = dist/(2*math.sqrt(2))
    tvecs_red = [[random.gauss(-mean_xy, sigma), random.gauss(-mean_xy, sigma), 'red'] for i in range(num_data_points//2)]
    tvecs_blue = [[random.gauss(mean_xy, sigma), random.gauss(mean_xy, sigma), 'blue'] for i in range(num_data_points//2, num_data_points)]
    cvecs = [[random.uniform(-(mean_xy+2*sigma), mean_xy+2*sigma), random.uniform(-(mean_xy+2*sigma), mean_xy+2*sigma), None] for i in range(num_class_points)]
#    print(cvecs)
#    print('')
    cvecs_fit = kNNclassification(tvecs_red + tvecs_blue, cvecs, [2], k, metric=euclideanDist)
    # prep feature vectors for plotting - split x and y components into separate lists based on color
    tvecs_red_x = [tvecs_red[i][0] for i in range(len(tvecs_red))]
    tvecs_red_y = [tvecs_red[i][1] for i in range(len(tvecs_red))]
    tvecs_blue_x = [tvecs_blue[i][0] for i in range(len(tvecs_blue))]
    tvecs_blue_y = [tvecs_blue[i][1] for i in range(len(tvecs_blue))]
    cvecs_red_x, cvecs_red_y = [], []
    cvecs_blue_x, cvecs_blue_y = [], []
    for vec in cvecs_fit:
        if vec[2] == 'red':
            cvecs_red_x.append(vec[0])
            cvecs_red_y.append(vec[1])
        elif vec[2] == 'blue':
            cvecs_blue_x.append(vec[0])
            cvecs_blue_y.append(vec[1])
#    print(cvecs_fit)
    plt.figure('scatter')
    plt.scatter(tvecs_red_x, tvecs_red_y, c='r', marker='.', label='Red training data')
    plt.scatter(cvecs_red_x, cvecs_red_y, c='r', marker='^', label='Red fit data')
    plt.scatter(tvecs_blue_x, tvecs_blue_y, c='b', marker='.', label='Blue training data')
    plt.scatter(cvecs_blue_x, cvecs_blue_y, c='b', marker='v', label='Blue fit data')
    plt.title('kNN classification of {} training points with k={}'.format(num_data_points, k))
    plt.show()
    

twoClassificationUnitTest(100, 100, k=3, dist=3, sigma=0.5)






