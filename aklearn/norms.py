'''Docstring for the norms.py module.

We define multiple notions of distance and norms to use in our 
model construction and evaluation.

'''

import numpy as np


def euclidean_2(x):
    return np.linalg.norm(x)


def euclidean_infty(x):
    return np.linalg.norm(x, ord=np.inf)


def euclidean_1(x):
    return np.linalg.norm(x, ord=1)

