'''This module builds a class for logistic regression problems.
We compute the solution by directly maximizing the log-likelihood.
To do, we implement the Newton-Raphson method, following the treatment
in the [1]_.

References
-----------
Hastie, T., Tibshirani, R., & Friedman, J. (2001). The Elements of Statistical Learning. Springer New York Inc..
'''

import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.optimize.optimize import _minimize_bfgs
from scipy.special import expit
from src.classification.classification import Classification
from src.helperfunctions.evaluation_metrics import evaluate_accuracy, confusion_matrix
from src.helperfunctions.preprocessing import scale_and_center


class Logistic(Classification):
    '''
    A class used to represent a logistic regression classifier.
    We only list non-inherited attributes. We use an optimizer
    from scipy to manually compute the maximum likelihood estimator.

    Parameters
    -----------
    features : numpy.ndarray
        Design matrix of explanatory variables.
    output : numpy.ndarray
        Labels of data corresponding to feature matrix.
    split_proportion : float
        Proportion of data to use for training; between 0 and 1.
    threshold : float
        The minimum probability needed to classify a datapoint as a 1.
    number_labels : int
        The number of labels present in the data.
    standardized : bool
        Whether to center/scale the data (train/test done separately).
        True by default.

    Attributes
    ----------
    tolerance : float
        One of two possible stopping criterion for the Newton-Raphson algorithm.
        It sets the required change in L_2 norm between successive coefficients.
        The algorithm will terminate no matter if the max_steps is reached.
    max_steps : int
        The maximum number of steps permitted for the Newton-Raphson algorithm.
        The algorithm terminates earlier if the tolerance is achieved, however,
    coefficients : numpy.ndarray
        The coefficients in the logistic regression model.
    threshold : float
        The minimum probability needed to classify a datapoint as a 1.
    train_probs : numpy.ndarray
        The predicted probabilities each training observation has label 1.
    train_predictions : numpy.ndarray
        The classified labels for the training data.
    test_probs : numpy.ndarray
        The predicted probabilities each test observation has label 1.
    test_predictions : numpy.ndarray
        The labels predicted for the given test data.
    train_accuracy : float
        The accuracy of the classifier evaluated on training data.
    test_accuracy : float
        The accuracy of the classifier evaluated on test data.
    train_confusion : float
        The accuracy of the classifier evaluated on training data.
    test_confusion : numpy.ndarray
        A confusion matrix of the classifier evaluated on test data.

    '''

    def __init__(self, features, output, split_proportion=0.75, threshold=0.5, 
                 number_labels=None, standardized=True, tolerance = 0.01,
                 max_steps = 500):
        if standardized:
            self.features = scale_and_center(features)

        # Add column for intercept
        self.features = np.append(np.ones((features.shape[0], 1)),
                                  features,
                                  axis=1)

        super().__init__(self.features, output, split_proportion, 
                         standardized=False)
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.coefficients = self.fit()
        self.threshold = threshold
        self.train_probs = Logistic.probability_estimate(self.train_features, 
                                            self.coefficients)
        self.train_predictions = np.array([1 if self.train_probs[i] > self.threshold else 0 for \
            i in range(self.train_size)])

        self.test_probs = Logistic.probability_estimate(self.test_features, 
                                           self.coefficients)
        self.test_predictions = np.array([1 if self.test_probs[i] > self.threshold else 0 for \
            i in range(self.test_size)])
        self.train_accuracy = evaluate_accuracy(self.train_predictions, 
                                                self.train_output)
        self.test_accuracy = evaluate_accuracy(self.test_predictions, 
                                               self.test_output)
        self.train_confusion = confusion_matrix(self.number_labels, 
                                                self.train_predictions,
                                                self.train_output)
        self.test_confusion = confusion_matrix(self.number_labels, 
                                               self.test_predictions, 
                                               self.test_output)
    @staticmethod
    def probability_estimate(features, coefficients):
        '''Compute P(y = 1|x, beta) = 1/(1+exp(-beta^Tx)).
        It can be used for both training and for prediction.
        
        Parameters
        -------------
        features : numpy.ndarray
            A design matrix of observations to condition on.
        coefficients : numpy.ndarray
            A vector of possible coefficients

        Returns
        -------
        estimate : float
            The estimated probability of an label of 1 for each observation,
            conditional on the data and coefficient.
        '''
        n = features.shape[0]
        dot_prods = features @ coefficients
        estimates = np.zeros(n)
        for i in range(n):
            if dot_prods[i] > 40:
                estimates[i] = 1
            elif dot_prods[i] < -40:
                estimates[i] = 0
            else:
                estimates[i] = 1 / (1 + np.exp(-dot_prods[i]))
        return estimates

    @staticmethod
    def weighted_matrix(features, coefficients):
        '''Compute weight matrix for the weighted least squares problem
        used in the Newton-Raphson algorithm of solving logistic regression.
        
        Parameters
        -------------
        features : numpy.ndarray
            A design matrix of observations (including all 1s column)
        coefficients : numpy.ndarray
            A vector of possible coefficients

        Returns
        -------
        wt_matrix : numpy.ndarray
            Diagonal matrix with ith entries p(1-p), where
            p = P(y = 1|X = x_i, beta).

        See Also
        ---------
        Logistic.newton_raphson_update : Implement a single step of the Newton-Raphson algorithm.
        '''
        probabilities = Logistic.probability_estimate(features, coefficients)
        wt_matrix_one = np.diag(probabilities)
       
        wt_matrix_two = np.diag(1 - probabilities)

        # Since diagonal, matrix multiplication is same as entry-wise multiplication 
        wt_matrix = wt_matrix_one @ wt_matrix_two  
       
        return wt_matrix
    
    @staticmethod
    def newton_raphson_update(features, beta_old, output):
        '''Compute a single step of the Newton-Raphson algorithm to compute
        the coefficient maximizing the likelihood for the logistic model.

        This algorithm is a root finder, i.e., finds a solution to f(x)=0
        for some function f. In this case, we set f to be the derivative of the
        log likelihood (i.e., the score function), to find the MLE. 

        Parameters
        -----------
        beta_old : numpy.ndarray
            The coefficients we would like to update
        features : numpy.ndarray
            A design matrix of explanatory variables
        output : numpy.ndarray
            The labels corresponding to our observed features

        Returns
        --------
        beta_new : numpy.ndarray
            The updated coefficients.
        '''
        X = features
        W = Logistic.weighted_matrix(features, beta_old)  # Weight matrix
        y = output
        probabilities = Logistic.probability_estimate(features, beta_old)
        beta_new = beta_old + np.linalg.pinv(X.T @ W @ X) @ X.T @ (y - probabilities)

        return beta_new

    def fit(self):
        '''
        Calculate the coefficient estimate on the training data.

        Returns
        --------
        beta_new : numpy.ndarray
            The estimated coefficients for logistic regression.
        '''

        beta_init = np.zeros(self.dimension)

        # Proportion of 1s in the data
        ratio = np.sum(self.train_output)/self.train_size

        # A recommended initial starting point for the intercept
        beta_init[0] = np.log(ratio/(1-ratio))

        # Guarantee we always make the first step
        coeff_change = np.Inf 

        # Keep track of the steps taken
        steps = 1

        while coeff_change > self.tolerance and steps < self.max_steps:
            beta_new = Logistic.newton_raphson_update(self.train_features, 
                                                      beta_init,
                                                      self.train_output)
            coeff_change = np.linalg.norm(beta_new - beta_init)
            beta_init = beta_new
            steps += 1

        return beta_new

