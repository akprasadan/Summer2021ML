'''This module builds a base class for classification problems, such as logistic
regression or k-nearest neighbors classification. 
The preprocessing (if applicable) is done at this class level.

'''

import numpy as np
from preprocessing import train_test_split, scale_and_center
from evaluation_metrics import evaluate_accuracy


class Classification:
    def __init__(self, features, output, split_proportion, number_labels=None, 
                 standardized=True):
        # Default procedure is to assume all labels appear in output
        # If labels are missing in data, specify number_labels manually
        if number_labels is None:  
            self.number_labels = len(np.unique(output)) 
        else: 
            self.number_labels = number_labels
        self.labels = np.arange()
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
