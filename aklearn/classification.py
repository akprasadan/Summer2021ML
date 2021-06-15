import numpy as np
from preprocessing import train_test_split, scale_and_center
from evaluation_metrics import evaluate_accuracy

class Classification:
    def __init__(self, features, output, split_proportion, number_labels = None, standardized = True):
        if number_labels is None: # Default procedure is to assume all labels appear in the output; otherwise, specify manually
            self.number_labels = len(np.unique(output)) 
        else: 
            self.number_labels = number_labels
        self.labels = np.arange()
        self.sample_size, \
        self.train_size, \
        self.test_size, \
        self.train_rows, \
        self.test_rows, \
        self.train_features, \
        self.test_features, \
        self.train_output, \
        self.test_output = train_test_split(features, output, split_proportion)
    
        if standardized:
            self.standardize()

    def standardize(self):
        self.train_features = scale_and_center(self.train_features)
        self.test_features = scale_and_center(self.test_features)
