import numpy as np
from collections import namedtuple

def scale_and_center(feature_matrix):
    '''Center and scale the feature matrix.'''

    n_rows, n_cols = feature_matrix.shape
    for column in range(n_cols):
        feature_matrix[:, column] = (feature_matrix[:, column] - np.mean(feature_matrix[:, column])*np.ones(n_rows)) \
            / np.std(feature_matrix[:, column])
    return feature_matrix

def train_test_split(features, output, split_proportion):
    sample_size = features.shape[0]
    train_size = np.ceil(sample_size * split_proportion)
    test_size = sample_size - train_size
    # https://stackoverflow.com/a/14262743
    train_rows = np.random.choice(train_size.shape[0], train_size, replace=False)
    test_rows = np.setdiff1d(np.arange(sample_size), train_rows)
    train_features = features[train_rows]
    test_features = features[test_rows]
    train_output = output[train_rows]
    test_output = output[test_rows]

    split_information = ['sample_size', 'train_size', 'test_size', 'train_rows', 'test_rows', \
        'train_features', 'test_features', 'train_output', 'test_output']
    TrainTestSplit = namedtuple('TrainTestSplit', split_information)
    split_values = TrainTestSplit(sample_size, train_size, test_size, train_rows, test_rows, \
        train_features, test_features, train_output, test_output)

    return split_values