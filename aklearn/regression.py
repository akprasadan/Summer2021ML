import numpy as np
#from aklearn.preprocessing import train_test_split, scale_and_center
from preprocessing import train_test_split, scale_and_center

class Regression:
    def __init__(self, features, output, split_proportion):
        self.sample_size, \
        self.train_size, \
        self.test_size, \
        self.train_rows, \
        self.test_rows, \
        self.train_features, \
        self.test_features, \
        self.train_output, \
        self.test_output = train_test_split(features, output, split_proportion)

    def finalize_preprocess(self):
        self.train_features = scale_and_center(self.train_features)
        self.test_features = scale_and_center(self.test_features)