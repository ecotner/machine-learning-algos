# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:51:19 2018

Testing a fanciful optimization algorithm that uses bisection within a trust region to find local minima rather than typical gradient descent.

The trust region is a multidimensional box within which the algorithm assumes there is some minimum. It is defined by a minimum and maximum value for each parameter to optimize. The algorithm then executes the following pseudocode:

requires: the function f(x) to be optimized, the dimensions of the trust region {x_i,min, x_i,max}, and some parameters \epsilon and \epsilon' for determining convergence
#initialize a point within the trust region (eg x_i = (x_i,max - x_i,min)/2, maybe even some random point)
initialize x_high = x_max and x_low = x_min
initialize y to some random value such that d(x,y) >> \epsilon
initialize fx, fy, to some random (very large) values
while |f'(x)| > \epsilon' (or maybe min(f'(x)) > \epsilon'):
    while d(x,y) > \epsilon:
        y = x
        fy = fx
        x = (x_high - x_low)/2
        fx, f'x = f(x), f'(x)
        if fx < fy:
            for i in range(len(x)):
                if f'x_i > 0:
                    x_i,high = x_i
                else:
                    x_i,low = x_i
        else:
            for i in range(len(x)):
                if x_i > y_i:
                    x_i,high = x_i
                else:
                    x_i,low = x_i
        

@author: Eric Cotner
"""

#import tensorflow as tf
import numpy as np

v1 = np.array([-0.299013, -0.797626, 0.886658, 0.587595, -0.477458, 0.893444, \
-0.0637873, -0.078859, 0.378553, -0.10268, -0.462661, 0.325949, \
-0.68949, 0.455915, 0.864663, 0.996786, -0.913053, 0.837327, \
0.902919, -0.944088, -0.357138, 0.579116, 0.960425, -0.180236, \
0.988199, 0.509655, -0.892106, 0.98814, 0.840945, 0.511892, 0.993576, \
-0.861147, 0.726407, -0.328817, 0.0494334, 0.459763, 0.525014, \
0.851661, -0.710177, -0.510915, -0.308796, 0.75938, -0.450828, \
-0.0850567, -0.381993, -0.467252, 0.838615, -0.566686, -0.515292, \
-0.0257465, 0.714141, -0.0838879, -0.488262, -0.636056, 0.877806, \
0.454976, -0.0279591, 0.769641, -0.554391, -0.319358, -0.00173732, \
0.751378, 0.0787349, -0.632725, 0.141494, -0.999384, -0.710652, \
-0.662459, 0.0532185, -0.403391, -0.440469, 0.936601, 0.598615, \
-0.916824, 0.174961, -0.221508, 0.115322, 0.983977, -0.951348, \
0.550263, 0.17719, 0.853228, 0.597095, 0.538372, 0.307375, 0.247225, \
0.038801, -0.254455, 0.486723, -0.269534, -0.551111, 0.232744, \
-0.469582, 0.033263, 0.156852, 0.453815, -0.992615, 0.079491, \
-0.996247, 0.0905674])
v2 = np.array([-0.435794, 0.467538, -0.760102, 0.294776, 0.95925, 0.780255, \
-0.214702, -0.731142, 0.370593, 0.357245, -0.375171, -0.0485401, \
0.446523, 0.135268, 0.926484, 0.634331, 0.515376, 0.436001, 0.775967, \
0.68646, 0.889052, -0.846963, 0.681764, -0.247893, 0.694644, \
0.310637, 0.487518, -0.356501, -0.849655, 0.552269, 0.85774, \
-0.39664, -0.0067022, -0.0298129, -0.924385, 0.450007, -0.877253, \
0.954481, 0.492704, 0.823752, 0.0952775, 0.861476, -0.834253, \
-0.668237, -0.524156, 0.00641007, 0.591181, -0.954494, -0.483252, \
-0.142982, 0.488174, 0.326234, -0.0520931, 0.197507, -0.775348, \
-0.471227, 0.444509, 0.283061, -0.910649, -0.360198, 0.769922, \
-0.569082, -0.543984, 0.454936, 0.131984, 0.058913, 0.364901, \
-0.853314, 0.839522, 0.552018, -0.820713, -0.70749, 0.47778, \
-0.222899, 0.127313, -0.501571, -0.694214, 0.722066, -0.779239, \
-0.726756, 0.878532, -0.697113, -0.302891, -0.0584644, -0.78662, \
0.607098, -0.798108, -0.41716, -0.303503, -0.132181, -0.989345, \
-0.751665, -0.761776, -0.430831, 0.807281, -0.0459047, 0.356306, \
0.452181, 0.739621, 0.844488])
x_true = np.array([-0.04441, -0.137214, 0.0627554, 0.0655251, -0.188308, -0.102436, \
-0.0103253, 0.0177914, 0.0346801, -0.107982, -0.0513226, 0.120307, \
-0.163588, 0.062118, -0.0946566, 0.0501803, -0.171972, 0.0710337, \
-0.0605558, -0.0404087, -0.17533, 0.239653, -0.0802945, -0.0362982, \
0.0450959, 0.107896, -0.157138, 0.187321, 0.0919886, -0.104461, \
-0.106079, -0.0740708, 0.110619, -0.100375, 0.165406, -0.0861797, \
0.0834241, -0.0215206, -0.0501289, -0.0922033, 0.0685867, -0.0788589, \
0.0301359, 0.0277926, -0.0407813, 0.0552473, 0.0611228, -0.00188017, \
-0.0673314, -0.0163331, -0.0412298, 0.0599385, 0.0615489, 0.0337295, \
0.0652766, 0.15843, -0.109407, -0.0198848, 0.145799, 0.0961396, \
0.00792616, 0.0760911, 0.00578952, -0.00281463, 0.0870401, -0.142195, \
-0.137407, -0.0248336, -0.0297983, -0.150887, 0.00604728, 0.216764, \
-0.069337, -0.0819212, 0.0570496, -0.0518714, 0.0393681, -0.1065, \
-0.0456613, 0.0706132, -0.149271, 0.0273005, 0.0555869, 0.115395, \
0.197279, 0.0278271, 0.19191, 0.103559, -0.0311027, -0.0836396, \
-0.00654159, 0.191216, 0.170309, 0.00524385, -0.00785229, 0.00432147, \
-0.16453, 0.0471375, -0.0667361, -0.166898])

def f(x):
#    return (np.sum(x**2)-1)**2 + np.dot(v,x)
    return (np.sum(x**2)-1)**2 + np.cos(10*np.dot(v1,x)) + np.cos(10*np.dot(v2,x))

def df(x):
        return 4*x*(np.sum(x**2)-1) - np.sin(10*np.dot(v1,x))*10*v1 - np.sin(10*np.dot(v2,x))*10*v2

def d(x,y, norm='inf'):
    if norm == 2:
        return np.sqrt(np.sum((x-y)**2))
    elif norm == 1:
        return np.sum(np.abs(x-y))
    elif norm == 'inf':
        return np.max(np.abs(x-y))
#    return np.sqrt(np.sum((x-y)**2))

epsilon = 1e-10
#x = np.array([-.5,-.3], dtype=float)
x = np.random.uniform(-2,2,len(v1))
trust_range = 2.0
global n
n = 0
def MultiDimensionalBisection(x, trust_range, epsilon=1e-6):
    global n
    m = 0
    x_prev = 10*x[:]
#    while trust_range > epsilon:
    #for cycle in range(100):
    while (np.max(np.abs((x/x_prev)-1)) > epsilon):
        m += 1
        x_high = x + (trust_range)*np.ones(len(x), dtype=float)
        x_low = x - (trust_range)*np.ones(len(x), dtype=float)
        x_prev = x[:]
        y = np.inf*np.ones(len(x), dtype=float)
        fy = np.inf
        fx_best = np.inf
        while d(x,y) > epsilon:
#        while np.max(np.abs((x/y)-1)) > epsilon:
            n += 1
            # Set y to value of x from previous step
            fx = f(x)
            dfx = df(x)
    #        print('x,y = {},{}'.format(x,y))
    #        print('x_high, x_low = {},{}\n'.format(x_high,x_low))
            if (fx <= fy):
                for i in range(len(x)):
                    if dfx[i] > 0:
                        x_high[i] = x[i]
                    else:
                        x_low[i] = x[i]
            else:
                for i in range(len(x)):
                    if x[i] > y[i]:
                        x_high[i] = x[i]
#                        assert x_low[i] == y[i]
                    else:
                        x_low[i] = x[i]
#                        assert x_high[i] == y[i]
            if fx < fx_best:
                x_best = x[:]
                fx_best = fx
            y = x[:]
            fy = fx
            x = (x_high + x_low)/2.
        # Need some way to update x so that it's the best value seen over the entire loop
        f_low = f(x_low)
        f_high = f(x_high)
        a = np.argmin([fx, fy, fx_best, f_low, f_high])
    #    print(a)
        x = [x, y, x_best, x_low, x_high][a]
        fx = [fx, fy, fx_best, f_low, f_high][a]
        trust_range = d(x,x_prev+(epsilon)*np.random.randn(len(x)))
    #    if fx <= f(x_prev):
    #        print(True)
    #    else:
    #        print(False)
#        print(d(x,x_prev))
    return x

x_best = np.random.uniform(-2,2,len(v1))
for i in range(100):
    x = np.random.uniform(-2,2,len(v1))
    x = MultiDimensionalBisection(x, 2, 1e-3)
    if f(x) < f(x_best):
        x_best = x[:]
    print('step {}'.format(i))

print('f(x_best) = {}, f(x_true) = {}, n_steps = {}'.format(f(x_best), f(x_true), n))

#print('x = {}\nfx = {:.2e}, true fx = {:.2e}, |dfx| = {:.2e}, error = {:.2e}, n_steps = {}, m_cycles = {}'.format(x, fx, f(x_true), np.linalg.norm(df(x)), d(x,x_true,2), n, m))









