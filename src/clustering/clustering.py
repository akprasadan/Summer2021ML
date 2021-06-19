'''
This module builds a a base class for clustering problems, such as k-means. The preprocessing (if applicable)
is done at this class level.

'''

import numpy as np
from src.helperfunctions.preprocessing import train_test_split, scale_and_center


class Clustering:
    """
    A class used to represent a clustering algorithm.

    Parameters
    -----------
    features : numpy.ndarray
        Design matrix of explanatory variables.
    standardized : bool
        Whether to center/scale the data. True by default.

    Attributes
    -----------
    sample_size : int
        The sample size of all given data (train and test).
    dimension : int
        The number of dimensions of the data, or columns of design matrix.
        Does not include output.

    """
    def __init__(self, features, standardized=True):
        self.features = features
        self.sample_size = features.shape[0]
        self.dimension = self.features.shape[1]

        if standardized:
            self.standardize()

    def standardize(self):
        '''
        Separately scale/center the train and test data so each feature
        (column of observations) has 0 mean and unit variance.
        '''
        self.train_features = scale_and_center(self.train_features)
        self.test_features = scale_and_center(self.test_features)
