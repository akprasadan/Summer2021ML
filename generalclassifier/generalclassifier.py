import numpy as np


class SupervisedAlgorithm:

    def __init__(self, features, output, split_proportion):
        self.features = np.array(features)
        self.output = np.array(output)
        self.split_proportion = split_proportion
        self.sample_size = self.features.shape[0]
        self.train_size = np.ceil(self.sample_size * split_proportion)
        # https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
        self.train_rows = np.random.choice(self.train_size.shape[0], self.train_size, replace=False)
        self.test_rows = np.setdiff1d(np.arange(self.sample_size), self.train_rows)
        self.train_features = self.features[self.train_rows]
        self.test_features = self.features[self.test_rows]
        self.train_output = self.output[self.train_rows]
        self.test_output = self.output[self.test_rows]
    
    def fit_to(self, input, output):
        pass
    


class Classification(SupervisedAlgorithm):

    def __init__(self, features, output, split_proportion):
        super().__init__(features, output, split_proportion)

    @staticmethod
    def evaluate_accuracy(predicted_output, true_output):
        number_predictions = predicted_output.shape[0]
        correct_predictions = np.count_nonzero(predicted_outupt == true_output)
        return correct_predictions / number_predictions
    
