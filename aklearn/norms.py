'''This module defines multiple notions of distance and norms to use in our 
model construction and evaluation.

'''

import numpy as np


def euclidean_2(x):
    '''Compute the Euclidean l_2 norm.'''
    return np.linalg.norm(x)


def euclidean_infty(x):
    '''Compute the Euclidean l_infinity norm.'''
    return np.linalg.norm(x, ord=np.inf)


def euclidean_1(x):
    '''Compute the Euclidean l_1 norm (Manhattan distance).'''
    return np.linalg.norm(x, ord=1)

