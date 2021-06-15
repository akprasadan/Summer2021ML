import numpy as np

def euclidean_2(x):
    return np.linalg.norm(x)

def euclidean_infty(x):
    return np.linalg.norm(x, ord = np.inf)

def euclidean_1(x):
    return np.linalg.norm(x, ord = 1)

