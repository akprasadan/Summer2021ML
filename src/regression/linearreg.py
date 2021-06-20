'''This module is for performing linear regression models.

'''
from src.helperfunctions.preprocessing import scale_and_center
import numpy as np
from src.regression.regression import Regression
from numpy.linalg import inv
from src.helperfunctions.evaluation_metrics import evaluate_regression_error


class Linear(Regression):
    '''
    A class used to represent a linear regression classifier.

    Parameters
    -----------
    features : numpy.ndarray
        Design matrix of explanatory variables, including vector of 1s in first column.
    output : numpy.ndarray
        Labels of data corresponding to feature matrix.
    split_proportion : float
        Proportion of data to use for training; between 0 and 1.
    standardized : bool
        Whether to center/scale the data (train/test done separately).
        True by default.

    Attributes
    ----------
    coefficients : numpy.ndarray
        The coefficients in the logistic regression model.
        The first coefficient is the intercept.
    train_predictions : numpy.ndarray
        The predicted output values for the training data.
    test_predictions : numpy.ndarray
        The predicted output values for the test data.
    train_error : float
        The error of model on training data (default is MSE).
    train_error : float
        The error of model on test data (default is MSE).

    '''
    def __init__(self, features, output, split_proportion=0.75,
                 standardized=True):
        if standardized:
            self.features = scale_and_center(features)
        # Add column for intercept
        self.features = np.append(np.ones((features.shape[0], 1)),
                                  features,
                                  axis=1)

        super().__init__(self.features, output, split_proportion, standardized=False)

        # First element is the intercept term
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
            Vector of coefficients of length self.dimension.
            The first element is the intercept term.
        '''
        train_X = self.train_features
        train_y = self.train_output
        coefficients = inv(train_X.T @ train_X) @ train_X.T @ train_y

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

        prediction = features @ coefficients
        return prediction

