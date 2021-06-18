'''This module provides data processing and preparation tools.

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

    Parameters
    ----------
    feature_matrix : numpy.ndarray
        Design matrix of explanatory variables.
    
    Returns
    -------
    feature_matrix : numpy.ndarray
        Design matrix of scaled and centered explanatory variables.
    '''

    n_rows, n_cols = feature_matrix.shape
    for column in range(n_cols):
        column_mean = np.mean(feature_matrix[:, column])*np.ones(n_rows)
        column_std_dev = np.std(feature_matrix[:, column])
        centered_column = feature_matrix[:, column] - column_mean
        centered_and_scaled_column = centered_column / column_std_dev

        feature_matrix[:, column] = centered_and_scaled_column
            
    return feature_matrix


def train_test_split(features, output, split_proportion):
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
    train_size = np.ceil(sample_size * split_proportion).astype(int)
    test_size = sample_size - train_size
    train_rows = np.random.choice(sample_size, train_size, 
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

def cross_validation_folds_idx(row_count, fold_count):
    '''Partition the (training) dataset into folds.

    Parameters
    -----------
    row_count : int
        Sample size of training data we form folds from.
    fold_count : int    
        The number of folds to produce; cannot exceed row_count.

    Raises
    -------
    AssertionError
        If more folds are requested than there are observations
    '''

    assert fold_count <= row_count, "There cannot be more folds than the sample size."
    rows_per_fold = row_count // fold_count 

    # Indices that have not been assigned to a fold yet; will be updated
    row_indices = np.arange(row_count) 

    # Array to store our folds: each row stores indices in that fold
    folds = np.zeroes((fold_count, rows_per_fold), dtype = np.int8)

    for fold in fold_count:
        fold_rows = np.random.choice(row_indices, size=rows_per_fold, replace=False)
        row_indices = np.setdiff1d(row_indices, fold_rows)
        folds[fold, :] = fold_rows
    
    return folds