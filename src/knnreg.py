'''This module builds a class for k-nearest neighbor classification.
'''

from knn_classify import KNNClassify
import numpy as np


class KNNRegression(KNNClassify):
    '''
    A class used to represent a k-nearest neighbor regressor.
    The regression methods and attributes can be found in the 
    KNNClassify class.

    Parameters
    -----------
    features : numpy.ndarray
        Design matrix of explanatory variables.
    output : numpy.ndarray
        Labels of data corresponding to feature matrix.
    split_proportion : float
        Proportion of data to use for training; between 0 and 1.
    standardized : bool
        Whether to center/scale the data (train/test done separately).
        True by default.
    k : int
        The number of neighbors to use in the algorithm.
    '''

    def __init__(self, features, output, split_proportion=0.75,
                standardized=True, k=3, classify = False):
        super().__init__(features, output, split_proportion, 
                         standardized)

