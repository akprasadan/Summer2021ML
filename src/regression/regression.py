'''
This module builds a a base class for regression problems, such as least squares
or k-nearest neighbors regressions. The preprocessing (if applicable)
is done at this class level.

'''

import numpy as np
from src.helperfunctions.preprocessing import train_test_split, scale_and_center


class Regression:
    """
    A class used to represent a regression algorithm.

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

    Attributes
    -----------
    sample_size : int
        The sample size of all given data (train and test).
    train_size : int
        The sample size of the training data.
    test_size : int
        The sample size of the test data.
    train_rows : numpy.ndarray
        The list of indices for the train split.
    test_rows : numpy.ndarray
        The list of indices for the test split.
    train_features : numpy.ndarray
        The train design matrix.
    test_features : numpy.ndarray
        The test design matrix.
    train_output : numpy.ndarray
        The train output data.
    test_output : numpy.ndarray
        The test output data.
    dimension : int
        The number of dimensions of the data, or columns of design matrix.
        Does not include output.

    """
    def __init__(self, features, output, split_proportion=0.75, standardized=True):
        train_test_split_data = train_test_split(features, output, split_proportion)

        self.sample_size = train_test_split_data.sample_size
        self.train_size = train_test_split_data.train_size
        self.test_size = train_test_split_data.test_size
        self.train_rows = train_test_split_data.train_rows
        self.test_rows = train_test_split_data.test_rows
        self.train_features = train_test_split_data.train_features
        self.test_features = train_test_split_data.test_features
        self.train_output = train_test_split_data.train_output
        self.test_output = train_test_split_data.test_output
        self.dimension = self.train_features.shape[1]

        if standardized:
            self.standardize()

    def standardize(self):
        '''
        Separately scale/center the train and test data so each feature
        (column of observations) has 0 mean and unit variance.
        '''
        self.train_features = scale_and_center(self.train_features)
        self.test_features = scale_and_center(self.test_features)


