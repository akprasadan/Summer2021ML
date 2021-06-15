import numpy as np
from regression import Regression
from numpy.linalg import inv
from numpy import transpose as trans
from evaluation_metrics import evaluate_regression_error

class Linear(Regression):
    def __init__(self, features, output, split_proportion, standardized = True):
        super().__init__(features, output, split_proportion, standardized)
        self.coefficients = self.fit()
        self.train_predictions = Linear.predict(self.train_features, self.coefficients)
        self.test_predictions = Linear.predict(self.test_features, self.coefficients)
        self.train_error = evaluate_regression_error(self.train_predictions, self.train_output)
        self.test_error = evaluate_regression_error(self.test_predictions, self.test_output)
    
    def fit(self):
        coefficients = inv(trans(self.train_features) @ self.train_features) @ trans(self.train_features) @ self.train_output
        return coefficients
    
    @staticmethod
    def predict(features, coefficients):
        prediction = trans(features) @ coefficients
        return prediction
    