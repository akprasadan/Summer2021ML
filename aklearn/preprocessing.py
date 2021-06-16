'''Docstring for the preprocessing.py module.

This module provides data processing and preparation tools
    - Scaling/centering data
    - Generating Test/Train splits
    - Generating Cross validation Folds (In Progress)
    - Generating Bootstrap resamples (In Progress)

'''

import numpy as np
from collections import namedtuple


def scale_and_center(feature_matrix):
    '''
    Center and scale each individual column of the feature matrix 
    to have 0 mean and unit variance.
    '''

    n_rows, n_cols = feature_matrix.shape
    for column in range(n_cols):
        column_mean = np.mean(feature_matrix[:, column])*np.ones(n_rows)
        column_std_dev = np.std(feature_matrix[:, column])
        centered_column = feature_matrix[:, column] - column_mean
        centered_and_scaled_column = centered_column / column_std_dev

        feature_matrix[:, column] = centered_and_scaled_column
            
    return feature_matrix


def train_test_split(features, output, split_proportion=0.25):
    '''Split the data into training and testing sets.

    Parameters
    ----------
    features : numpy.ndarray
        Design matrix of explanatory variables
    output : numpy.ndarray
        The given response variables
    split_proportion : float
        The proportion of data used for training. Default is 25%.
        Must lie between 0 and 1.

    Returns
    -------
    split_values : namedTuple
        Stores the following values: ['sample_size', 'train_size', 
                         'test_size','train_rows', 'test_rows', 
                         'train_features', 'test_features', 
                         'train_output', 'test_output']

    Notes
    ------
    I used [2]_ to figure out how to randomly choose rows of an array.

    References
    ----------
    .. [2] https://stackoverflow.com/a/14262743
    '''

    sample_size = features.shape[0]
    train_size = np.ceil(sample_size * split_proportion)
    test_size = sample_size - train_size
    #
    train_rows = np.random.choice(train_size.shape[0], train_size, 
                                  replace=False)
    test_rows = np.setdiff1d(np.arange(sample_size), train_rows)
    train_features = features[train_rows]
    test_features = features[test_rows]
    train_output = output[train_rows]
    test_output = output[test_rows]

    split_information = ['sample_size', 'train_size', 'test_size', 
                         'train_rows', 'test_rows', 'train_features',
                         'test_features', 'train_output', 'test_output']

    TrainTestSplit = namedtuple('TrainTestSplit', split_information)

    split_values = TrainTestSplit(sample_size, train_size, test_size, 
                                  train_rows, test_rows, 
                                  train_features, test_features, 
                                  train_output, test_output)

    return split_values
