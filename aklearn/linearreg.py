import numpy as np
from regression import Regression
from numpy.linalg import inv
from numpy import transpose as trans
from evaluation_metrics import evaluate_accuracy, evaluate_mse

class Linear(Regression):
    def __init__(self, features, output, split_proportion):
        super().__init__(features, output, split_proportion)
        self.coefficients = self.fit()
        self.train_error()
        self.test_error()
    
    def fit(self):
        coefficients = inv(trans(self.train_features) @ self.train_features) @ trans(self.train_features) @ self.train_output
        return coefficients
    
    @staticmethod
    def predict(features, coefficients):
        prediction = trans(features) @ coefficients
        return prediction
    
    def train_error(self):
        train_mse = evaluate_mse(Linear.predict(self.train_features, self.coefficients), self.train_output)
        self.train_mse = train_mse

    def test_error(self):
        test_mse = evaluate_mse(Linear.predict(self.test_features, self.coefficients), self.test_output)
        self.test_mse = test_mse
