'''This module builds a class for logistic regression problems.
We compute the solution by directly maximizing the log-likelihood.
We use an existing software implementation to globally maximize the 
likelihood function: BFGS, available in 
scipy.optimize.minimize(method = 'BFGS')

'''

import numpy as np
from scipy.optimize import minimize
from scipy.optimize.optimize import _minimize_bfgs
from scipy.special import expit
from src.classification.classification import Classification
from src.helperfunctions.evaluation_metrics import evaluate_accuracy, confusion_matrix


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
                 number_labels=None, standardized=True):
        super().__init__(features, output, split_proportion, number_labels, 
                         standardized)
        self.coefficients = self.fit()
        print(self.coefficients)
        self.threshold = threshold
        self.train_probs = Logistic.predict(self.train_features, 
                                            self.coefficients)
        self.train_predictions = self.train_probs[self.train_probs 
                                                  >= self.threshold]
        self.test_probs = Logistic.predict(self.test_features, 
                                           self.coefficients)
        self.test_predictions = self.test_probs[self.test_probs 
                                                >= self.threshold]
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
    def logsumexp(x, y):
        '''Implements the log-sum-exp (LSE) function using a numerically stable formulation.

        Parameters
        -----------
        x : float
            An input to the LSE function
        y : float
            An input to the LSE function

        Returns
        --------
        lse : float
            The evaluation of LSE at x and y

        Notes
        ------
        The LSE function provides an alternative to a numerically unstable calculation
        relying on the sigmoid function f(x) = -log(1+exp(-x)), as exp(-x) may blow up.
        Instead, we use LSE(x, y) = log(exp(x)+exp(y)) and the fact that
        f(x) = -LSE(0, -x). A full explanation is given here _[5]. We then
        use a numerically preferable version of the LSE, given in _[6]. 

        References
        -----------
        .. [5] https://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
        .. [6] https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    '''
    
        z = np.array([x, y])
        max_input = np.max(z)

        # The quantity z_diff can be exponentiated with much less difficulty.
        z_diff = z - max_input
        
        exp_diff = np.exp(z_diff)
        lse = max_input + np.log(np.sum(exp_diff))

        return lse


    @staticmethod
    def lse_alternative(features, labels, coefficient):
        '''Compute log MLE for each coefficient using log sum exponential.

        Parameters
        ----------
        features : numpy.ndarray
            Design matrix of explanatory variables
        labels : numpy.ndarray
            The given output labels
        coefficients : numpy.ndarray
            Vector of coefficients for logistic regression.
        
        Returns
        -------
        loglikelihood : float
            The value of the log sigmoid with given data and coefficient.

        Notes
        ------
        We use the LSE for numerical ease (see Logistic.logsumexp for references).
        The mathematical formulas are given in _[5].
        '''
        lse_terms = np.zeros((features.shape[0]))
        for row in range(features.shape[0]):
            x_beta = np.dot(features[row, :], coefficient)
            if labels[row] == 1:
                print(Logistic.logsumexp(0, -x_beta))
                lse_terms[row] = np.log(Logistic.logsumexp(0, -x_beta))
            elif labels[row] == 0:
                print(Logistic.logsumexp(0, x_beta))
                lse_terms[row] = np.log(Logistic.logsumexp(0, x_beta)) 

        loglikelihood = np.sum(lse_terms)

        return loglikelihood

    @staticmethod
    def mle_finder(features, labels):
        '''
        Find the MLE for a logistic model; return the coefficient.

        Parameters
        ----------
        features : numpy.ndarray
            Design matrix of explanatory variables
        labels : numpy.ndarray
            The given output labels

        Returns
        -------
        mle : float
            The coefficient that solves the logistic regression problem for 
            the given data.

        Notes
        -----
        We use the BFGS algorithm as implemented in scipy to maximize
        the log-likelihood.
        This requires negating the log likelihood as we use minimization.

        See Also
        --------
        scipy.optimize.minimize
        '''

        def negative_log_likelihood(coefficient):
            ''' The negative log likelihood function that we minimize.
            '''
            value = -Logistic.lse_alternative(features, labels, 
                                            coefficient)
            return value

        # Initialize a coefficient to begin the maximization at
        coeff_init = np.ones((features.shape[1]), dtype = np.int8)
        optimum = _minimize_bfgs(negative_log_likelihood, coeff_init)
        mle = optimum.x  # Obtain argmin
        
        return mle

    def fit(self):
        '''
        Calculate the coefficient estimate on the training data.
        '''

        coefficients = Logistic.mle_finder(self.train_features, 
                                           self.train_output)

        return coefficients

    def predict(features, coefficients):
        '''Compute estimated output P(Y =1|X=x, beta_hat) 
           of logistic regression.

        Parameters
        ----------
        features : numpy.ndarray
            Design matrix of explanatory variables.
        coefficients : numpy.ndarray
            Vector of coefficients for logistic regression solution.

        Returns
        -------
        p : numpy.ndarray
            Predicted output (probability) for each observation.
        '''
        coeff_matrix = np.tile(coefficients, (features.shape[0], 1))
        dot_prods = np.sum(features * coeff_matrix, axis=1)
        p = expit(dot_prods)

        return p
    

X = np.random.rand(5,2)
y = np.random.randint(2, size=5)
from scipy.special import logsumexp



test = Logistic(X, y, standardized=False)




