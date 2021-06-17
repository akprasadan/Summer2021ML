'''This module builds a class for k-nearest neighbor classification.
'''

import numpy as np
from scipy.stats import mode
from classification import Classification
from evaluation_metrics import evaluate_accuracy, confusion_matrix
from numba import jit

class KNNClassify(Classification):
    '''
    A class used to represent a k-nearest neighbor classifier.
    We only list non-inherited attributes.

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
    
    Attributes
    ----------
    k : int
        The number of neighbors to use in the algorithm.
    test_predictions : numpy.ndarray
        The labels predicted for the given test data.
    test_accuracy : float
        The accuracy of the classifier evaluated on test data.
    test_confusion : numpy.ndarray
        A confusion matrix of the classifier evaluated on test data.

    Methods
    --------
    k_neighbors_idx
        Identify the k-nearest neighbors.
    classify_point
        Classify a datapoint given its k-nearest neighbors.
    predict
        Classify many test datapoints using some training data.
    '''
    def __init__(self, features, output, split_proportion,
                 number_labels=None, standardized=True, k=3):
        super().__init__(features, output, split_proportion, number_labels, 
                         standardized)
        self.k = k
        self.test_predictions = KNNClassify.predict(self.train_features,
                                                    self.train_output,
                                                    self.test_features,
                                                    self.k)
        self.test_accuracy = evaluate_accuracy(self.test_predictions, 
                                               self.test_output)
        self.test_confusion = confusion_matrix(self.number_labels, 
                                               self.test_predictions, 
                                               self.test_output)
    @staticmethod
    @jit(nopython=True)
    def k_neighbors_idx(feature_matrix, current_location, k):
        '''Find row indices (in given data) of the k closest neighbors 
        to a given data point.
        
        Parameters
        -----------
        feature_matrix : numpy.ndarray 
            Design matrix of explanatory variables
        current_location : numpy.ndarray
            Point we would like to classify, using its neighbors.
        k : int
            The number of neighbors to use.

        Returns
        --------
        k_nearest_idx : numpy.ndarray
            The k indices of the feature_matrix observations closest
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

        pairwise_differences = feature_matrix[:, None, :] - current_location[None, :, :]
        distance_matrix = np.linalg.norm(pairwise_differences)
        
        k_nearest_idx = np.argsort(distance_matrix)[:k]

        return k_nearest_idx
    
    @staticmethod
    @jit(nopython=True)
    def classify_point(feature_matrix, output, current_location, k):
        '''Classify a new datapoint based on its k neighbors.
        
        Parameters
        -----------
        feature_matrix : numpy.ndarray 
            Design matrix of explanatory variables.
        output : numpy.ndarray
            Labels corresponding to feature_matrix.
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
        '''

        k_nearest_idx = KNNClassify.k_neighbors_idx(feature_matrix, current_location, k)
        nearest_k_labels = output[k_nearest_idx, :]
        label_mode = mode(nearest_k_labels)[0]

        return label_mode

    @jit(nopython=True)
    def predict(train_features, train_output, test_features, k):
        '''Classify many new datapoints based on their k neighbors.
        
        Parameters
        -----------
        train_features : numpy.ndarray 
            Design matrix of explanatory variables.
        train_output : numpy.ndarray
            Labels corresponding to feature_matrix.
        test_features : numpy.ndarray
            Points we would like to classify, using their neighbors.
        k : int
            The number of neighbors to use.

        Returns
        --------
        test_labels : numpy.ndarray
            The predicted labels for each test datapoint.
        '''
        test_sample_size = test_features.shape[0]
        test_labels = np.zeros((test_sample_size, 1))

        for row in range(test_sample_size):
            test_labels[row] = KNNClassify.classify_point(train_features, 
                                                           train_output,
                                                           test_features[i, :],
                                                           k)
        return test_labels
        



    

