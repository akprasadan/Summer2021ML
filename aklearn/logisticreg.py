import numpy as np
from scipy.optimize import minimize
from scipy.optimize.optimize import _minimize_bfgs
from classification import Classification
from evaluation_metrics import evaluate_accuracy, confusion_matrix

class Logistic(Classification):
    def __init__(self, features, output, split_proportion, number_labels = None, standardized = True):
        super().__init__(features, output, split_proportion, number_labels, standardized)
        self.coefficients = self.fit()
        self.train_predictions = Logistic.predict(self.train_features, self.coefficients)
        self.test_predictions = Logistic.predict(self.test_features, self.coefficients)
        self.train_accuracy = evaluate_accuracy(self.train_predictions, self.train_output)
        self.test_accuracy = evaluate_accuracy(self.test_predictions, self.test_output)
        self.train_confusion = confusion_matrix(self.number_labels, self.train_predictions, \
            self.train_output)
        self.test_confusion = confusion_matrix(self.number_labels, self.test_predictions, \
            self.test_output)

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
    

