import numpy as np
from scipy.optimize import minimize
from scipy.optimize.optimize import _minimize_bfgs
from classification import Classification

class Logistic(Classification):
    def __init__(self, features, output, split_proportion):
        super().__init__(features, output, split_proportion)
        self.coefficients = self.fit()
        self.train_error()
        self.test_error()

    @staticmethod
    def loglikelihood(features, labels, coefficient):
        coeff_matrix = np.tile(coefficient, (features.shape[0], 1))
        dot_prods = np.sum(features * coeff_matrix, axis = 1)
        log_h = np.log(1/(1+np.exp(-dot_prods)))
        summands = np.multiply(log_h, labels)+np.multiply(1-log_h, 1-labels)
        return np.mean(summands)

    @staticmethod
    def mle_finder(features, labels):
        f = lambda coefficient: Logistic.loglikelihood(labels, labels, coefficient)
        optimum = _minimize_bfgs(f, np.ones(labels.shape[1]))
        return optimum.x

    def fit(self):
        coefficients = Logistic.mle_finder(self.train_features, self.train_output)
        return coefficients

    def predict(features, coefficients):
        coeff_matrix = np.tile(coefficients, (features.shape[0], 1))
        dot_prods = np.sum(features * coeff_matrix, axis = 1)
        h = 1/(1+np.exp(-dot_prods))
        return h
    
    def train_error(self):
        train_mse = evaluate_mse(Logistic.predict(self.train_features, self.coefficients), self.train_output)
        self.train_mse = train_mse

    def test_error(self):
        test_mse = evaluate_mse(Logistic.predict(self.test_features, self.coefficients), self.test_output)
        self.test_mse = test_mse




