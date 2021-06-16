'''Docstring for the linearreg.py module.

This module is for performing linear regression models.

'''

import numpy as np
from regression import Regression
from numpy.linalg import inv
from numpy import transpose as trans
from evaluation_metrics import evaluate_regression_error


class Linear(Regression):
    def __init__(self, features, output, split_proportion, 
                 standardized=True):
        super().__init__(features, output, split_proportion, standardized)
        self.coefficients = self.fit()
        self.train_predictions = Linear.predict(self.train_features, 
                                                self.coefficients)
        self.test_predictions = Linear.predict(self.test_features, 
                                               self.coefficients)
        self.train_error = evaluate_regression_error(self.train_predictions, 
                                                     self.train_output)
        self.test_error = evaluate_regression_error(self.test_predictions, 
                                                    self.test_output)
    
    def fit(self):
        '''Calculate the coefficient solving the least squares problem 
        using training data.

        Returns
        -------
        coefficients : numpy.ndarray
            Vector of coefficients of length self.dimension
        '''
        train_X = self.train_features
        train_y = self.train_output
        coefficients = inv(trans(train_X) @ train_X) @ trans(train_X) @ train_y

        return coefficients
    
    @staticmethod
    def predict(features, coefficients):
        '''Compute estimated output y = X*beta_hat of linear regression.

        Parameters
        ----------
        features : numpy.ndarray
            Design matrix of explanatory variables.
        coefficients : numpy.ndarray
            Vector of coefficients for least squares solution.

        Returns
        -------
        prediction : numpy.ndarray
            Predicted output for each observation.
        '''

        prediction = trans(features) @ coefficients
        return prediction
    