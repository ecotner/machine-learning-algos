# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:54:56 2017

Clustering algorithm which uses gravitational mechanics as inspiration. Has the advantage that the number of clusters does not need to be specified a priori, and it automatically builds a hierarchy of the strongest possible clusters.

Each feature vector \vec{x}_i is represented as a particle in Euclidean space which has a specified mass m_i. Each particle initially represents its own class, so there are as many classes as there are particles (N of them). At each time step, every particle i experiences a force due to particle j of

\vec{F}_{ij} = \gamma m_i m_j \hat{d}_{ij}/d_{ij}^n,

where \vec{d}_{ij} = \vec{x}_j - \vec{x}_i is the displacement vector going from particle i to particle j, \gamma is some coefficient of the gravitational strength (i.e. the learning rate), and n is some constant which determines the long and short range behavior of the gravitational force (n=2 corresponds to realistic Newtonian gravity). The forces on each particle i are summed at each timestep, and the feature vectors are updated as

\vec{x}_i <-- \vec{x}_i + \sum_{j!=i} \vec{F}_{ij}

One may also add momentum, but then it is possible to run into (pseudo-)stable bound orbits or grazing incidences which will slow the merger rate. We want the clusters to merge together as quickly as possible without moving too far from their original positions.

In order to prevent instability, \gamma can be adapted at each time step so that the maximum distance a particle travels is given by some other parameter \delta. This means that \gamma is scaled such that

\delta = max_i(abs(\vec{F}_i)) = \gamma*max_i(abs(\sum_{j!=i} m_i m_j \hat{d}_{ij}/d_{ij}^n)),

or, eliminating \gamma entirely, the feature update rule becomes

\vec{x}_i <-- \vec{x}_i + \delta \vec{\alpha}_i/\alpha_{max}, where
\vec{\alpha}_i = \sum_{j!=i} m_i m_j \hat{d}_{ij}/d_{ij}^n and
\alpha_{max} = max_i |\alpha_i|

After each update, we compute the distance matrix d_{ij} (an antisymmetric matrix since d_{ij} = -d{ji}), and if any element is less than some impact radius R, the two particles are merged into one particle k described by

\vec{x}_k = (m_i \vec{x}_i + m_j \vec{x}_j) / m_k, with
m_k = m_i + m_j

This particle will remember the labels of the original two particles. Then, through successive mergers over future timesteps, the number of particles will eventually be reduced to one. Along the way, we can construct a merger tree which contains the past history of the merger process (perhaps )


@author: Eric Cotner
"""






































