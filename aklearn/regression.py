'''Docstring for regression.py module.

This module builds a class for regression problems, such as least squares
or k-nearest neighbors regressions. The preprocessing (if applicable)
is done at this class level.

'''

import numpy as np
from preprocessing import train_test_split, scale_and_center


class Regression:
    def __init__(self, features, output, split_proportion, standardized=True):
        self.sample_size, 
        self.train_size, 
        self.test_size, 
        self.train_rows, 
        self.test_rows, 
        self.train_features, 
        self.test_features, 
        self.train_output, 
        self.test_output = train_test_split(features, output, split_proportion)
        self.dimension = self.train_rows.shape[1]

        if standardized:
            self.standardize()

    def standardize(self):
        '''
        Separately scale/center the train and test data so each feature
        (column of observations) has 0 mean and unit variance.
        '''
        self.train_features = scale_and_center(self.train_features)
        self.test_features = scale_and_center(self.test_features)
