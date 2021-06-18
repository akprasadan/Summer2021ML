'''This module builds a class for k-nearest neighbor classification.
'''

import numpy as np
from scipy.stats import mode
from classification import Classification
from evaluation_metrics import evaluate_accuracy, confusion_matrix, evaluate_regression_error
from numba import jit

class KNNClassify(Classification):
    '''
    A class used to represent a k-nearest neighbor classifier.
    We only list non-inherited attributes. We include regression
    functionality as well.

    Parameters
    -----------
    features : numpy.ndarray
        Design matrix of explanatory variables.
    output : numpy.ndarray
        Labels of data corresponding to feature matrix.
    split_proportion : float
        Proportion of data to use for training; between 0 and 1.
    number_labels : int
        The number of labels present in the data.
    standardized : bool
        Whether to center/scale the data (train/test done separately).
        True by default.
    k : int
        The number of neighbors to use in the algorithm.
    classify : bool
        Whether we are using this class for classification or regression.
        True by default. We will use instants with classify == False
        for a KNNRegression class.
    
    Attributes
    ----------
    k : int
        The number of neighbors to use in the algorithm.
    test_predictions : numpy.ndarray
        The labels predicted for the given test data (for classification).
    test_accuracy : float
        The accuracy of the classifier evaluated on test data 
        (for classification).
    test_confusion : numpy.ndarray
        A confusion matrix of the classifier evaluated on test data 
        (for classification).
    test_predictions_reg : numpy.ndarray
        The predicted output on test data (for regression).
    test_error : float
        The test MSE of model fit using training data (for regression).


    See Also
    ---------
    knnregression.KNNRegression : Class for a regression k-nearest neighbor model.
    '''

    def __init__(self, features, output, split_proportion,
                 number_labels=None, standardized=True, k=3, 
                 classify=True):
        super().__init__(features, output, split_proportion, number_labels, 
                         standardized)
        self.k = k
        if classify == True:
            self.test_predictions = KNNClassify.predict_class(self.train_features,
                                                        self.train_output,
                                                        self.test_features,
                                                        self.k)
            self.test_accuracy = evaluate_accuracy(self.test_predictions, 
                                               self.test_output)
            self.test_confusion = confusion_matrix(self.number_labels, 
                                               self.test_predictions, 
                                               self.test_output)
        else:
            self.test_predictions_reg = KNNClassify.predict_value(self.train_features,
                                                        self.train_output,
                                                        self.test_features,
                                                        self.k)
            self.test_error = evaluate_regression_error(self.test_predictions_reg, 
                                                        self.test_output)

    @staticmethod
    @jit(nopython=True)
    def k_neighbors_idx(features, current_location, k):
        '''Find row indices (in given data) of the k closest neighbors 
        to a given data point.
        
        Parameters
        -----------
        features : numpy.ndarray 
            Design matrix of explanatory variables
        current_location : numpy.ndarray
            Point we would like to classify, using its neighbors.
        k : int
            The number of neighbors to use.

        Returns
        --------
        k_nearest_idx : numpy.ndarray
            The k indices of the features observations closest
            to the current point.

        Notes
        ------
        An efficient numpy procedure (using its broadcasting functionality) to compute 
        all pairwise differences between two collections of data points is given in [1]_.
        We use this, as an alternative to using a manual nested 'for-loop' procedure.

        References
        ------------
        .. [1] https://sparrow.dev/pairwise-distance-in-numpy/
        '''

        pairwise_differences = features[:, None, :] - current_location[None, :, :]
        distance_matrix = np.linalg.norm(pairwise_differences)
        
        k_nearest_idx = np.argsort(distance_matrix)[:k]

        return k_nearest_idx
    
    @staticmethod
    @jit(nopython=True)
    def classify_point(features, output, current_location, k):
        '''Classify a new datapoint based on its k neighbors.
        
        Parameters
        -----------
        features : numpy.ndarray 
            Design matrix of explanatory variables.
        output : numpy.ndarray
            Labels corresponding to features.
        current_location : numpy.ndarray
            Point we would like to classify, using its neighbors.
        k : int
            The number of neighbors to use.

        Returns
        --------
        label_mode : int
            The predicted label (mode of labels of the k-nearest neighbors).

        Notes
        ------
        We choose the smallest label by default.

        See Also
        ---------
        KNNClassify.estimate_point : Find average output value among neighbors instead 
                                     of most common label (for regression).
        '''

        k_nearest_idx = KNNClassify.k_neighbors_idx(features, current_location, k)
        nearest_k_labels = output[k_nearest_idx, :]
        label_mode = mode(nearest_k_labels)[0]

        return label_mode

    @staticmethod
    @jit(nopython=True)
    def estimate_point(features, output, current_location, k):
        '''Estimate (for a regression context) a new datapoint based on its k neighbors.
        
        Parameters
        -----------
        features : numpy.ndarray 
            Design matrix of explanatory variables.
        output : numpy.ndarray
            Labels corresponding to features.
        current_location : numpy.ndarray
            Point we would like to classify, using its neighbors.
        k : int
            The number of neighbors to use.

        Returns
        --------
        output_estimate : int
            The predicted output value of the current location.

        See Also
        ---------
        KNNClassify.classify_point : Find most common label among neighbors instead of
                                     average output value (for classification).
        '''

        k_nearest_idx = KNNClassify.k_neighbors_idx(features, current_location, k)
        output_estimate = np.mean(output[k_nearest_idx, :])

        return output_estimate


    @jit(nopython=True)
    def predict_class(train_features, train_output, test_features, k):
        '''Classify many new datapoints based on their k neighbors.
        
        Parameters
        -----------
        train_features : numpy.ndarray 
            Design matrix of explanatory variables.
        train_output : numpy.ndarray
            Labels corresponding to features.
        test_features : numpy.ndarray
            Points we would like to classify, using their neighbors.
        k : int
            The number of neighbors to use.

        Returns
        --------
        test_labels : numpy.ndarray
            The predicted labels for each test datapoint.
        See Also
        ---------
        KNNClassify.predict_value : Predict output value instead 
                                     of label (for regression).
        '''
        test_sample_size = test_features.shape[0]
        test_labels = np.zeros((test_sample_size, 1), dtype = np.int8)

        for row in range(test_sample_size):
            test_labels[row] = KNNClassify.classify_point(train_features, 
                                                           train_output,
                                                           test_features[i, :],
                                                           k)
        return test_labels

    @jit(nopython=True)
    def predict_value(train_features, train_output, test_features, k):
        '''Classify many new datapoints based on their k neighbors.
        
        Parameters
        -----------
        train_features : numpy.ndarray 
            Design matrix of explanatory variables.
        train_output : numpy.ndarray
            Labels corresponding to features.
        test_features : numpy.ndarray
            Points we would like to classify, using their neighbors.
        k : int
            The number of neighbors to use.

        Returns
        --------
        test_estimates : numpy.ndarray
            The predicted output for each test datapoint.

        See Also
        ---------
        KNNClassify.predict_class : Predict label instead of output 
                                    value (for classification).
        '''
        test_sample_size = test_features.shape[0]
        test_estimates = np.zeros((test_sample_size, 1))

        for row in range(test_sample_size):
            test_estimates[row] = KNNClassify.estimate_point(train_features, 
                                                           train_output,
                                                           test_features[i, :],
                                                           k)
        return test_estimates
        



    

