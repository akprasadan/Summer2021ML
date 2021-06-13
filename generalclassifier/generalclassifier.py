import numpy as np
from numba import jit

class SupervisedAlgorithm:

    def __init__(self, features, output, split_proportion):
        self.features = np.array(features)
        self.output = np.array(output)
        self.split_proportion = split_proportion
        self.sample_size = self.features.shape[0]
        self.train_size = np.ceil(self.sample_size * split_proportion)
        # https://stackoverflow.com/a/14262743
        self.train_rows = np.random.choice(self.train_size.shape[0], self.train_size, replace=False)
        self.test_rows = np.setdiff1d(np.arange(self.sample_size), self.train_rows)
        self.train_features = self.features[self.train_rows]
        self.test_features = self.features[self.test_rows]
        self.train_output = self.output[self.train_rows]
        self.test_output = self.output[self.test_rows]
    
    def fit_to(self, input, output):
        pass
    
    @staticmethod
    def scale_and_center(feature_matrix):
        '''Center and scale the feature matrix.'''

        n_rows, n_cols = feature_matrix.shape
        for column in range(n_cols):
            feature_matrix[:, column] = (feature_matrix[:, column] - np.mean(feature_matrix[:, column])*np.ones(n_rows)) \
                / np.std(feature_matrix[:, column])
        return feature_matrix


class Classification(SupervisedAlgorithm):

    def __init__(self, features, output, split_proportion, number_labels = None):
        super().__init__(features, output, split_proportion)
        if number_labels is None: # Default procedure is to assume all labels appear in the output; otherwise, specify manually
            self.number_labels = len(np.unique(output)) 
        else: 
            self.number_labels = number_labels
        self.labels = np.arange()


    @staticmethod
    def evaluate_accuracy(predicted_output, true_output):
        number_predictions = predicted_output.shape[0]
        correct_predictions = np.count_nonzero(predicted_output == true_output)
        return correct_predictions / number_predictions
    
    @jit(nopython=True) 
    def confusion_matrix(self, predicted_output, true_output):
        confusion_matrix = np.zeros(shape = (self.number_labels, self.number_labels))
        output_combined = np.stack((true_output, predicted_output), axis = 1)  # each row has 2 elements

        for row_index in range(self.number_labels): #i
            for col_index in range(self.number_labels): #j
                # Let's compute the number of rows of output_combined containing (i,j) 
                # want predicted == j, true == i
                # Source of the following line's logic: https://stackoverflow.com/a/40382459
                confusion_matrix[row_index, col_index] = (output_combined == (row_index, col_index)).all(axis = 1).sum()

        return confusion_matrix

class Regression(SupervisedAlgorithm):

    def __init__(self, features, output, split_proportion, number_labels = None):
        super().__init__(features, output, split_proportion)

    @staticmethod
    def evaluate_mse(predicted_output, true_output):
        return np.linalg.norm(predicted_output, true_output)**2