'''This module builds a class for k-nearest neighbor classification.
'''

import numpy as np
from classification import Classification
from evaluation_metrics import evaluate_accuracy, confusion_matrix
from numba import jit

class QDA(Classification):
    '''
    A class used to represent a quadratic discriminant analysis classifier.
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
    
    Attributes
    ----------
    priors : list
        The estimated probability of being in each class
    features_subsets : numpy.ndarray
        A 3D array where for each i, features_subsets[:, :, i] is the rows of the training data
        with class i.
    class_feature_covs : numpy.ndarray
        A 3D array where for each i, class_feature_covs[:, :, i] is the covariance matrix of
        the rows of training data with class i.
    feature_means : numpy.ndarray
        A 2D array where for each i, feature_means[:, i] is the mean 
        of the rows of training data with class i.
    train_prediction : numpy.ndarray
        The labels predicted for the given test data (for classification).
    test_predictions : numpy.ndarray
        The labels predicted for the given test data (for classification).
    train_accuracy : float
        The accuracy of the classifier evaluated on test data
    test_accuracy : float
        The accuracy of the classifier evaluated on test data 
        (for classification).
    train_confusion : numpy.ndarray
        A confusion matrix of the classifier evaluated on training data 
        (for classification).
    test_confusion : numpy.ndarray
        A confusion matrix of the classifier evaluated on test data 
        (for classification).

    See Also
    ---------
    lda.LDA : Use the more restrictive linear discriminant analysis
    '''

    def __init__(self, features, output, split_proportion,
                 number_labels=None, standardized=True):
        super().__init__(features, output, split_proportion, number_labels, 
                         standardized)
        self.priors = [QDA.prior(self.train_output, i) for i in range(number_labels)]
        self.features_subsets = np.stack([self.train_features[output == k] for k in range(np.number)], axis = -1)
        self.class_feature_covs = np.stack([QDA.class_covariance(self.features_subsets[:,:, k]) for k in range(number_labels)], axis = -1)
        self.feature_means = np.mean(self.features_subsets, axis = 1)
        self.train_predictions = self.predict_many(self.train_features)
        self.test_predictions = self.predict_many(self.test_features)
        self.train_accuracy = evaluate_accuracy(self.train_predictions, 
                                            self.train_output)
        self.train_confusion = confusion_matrix(self.number_labels, 
                                            self.train_predictions, 
                                            self.train_output)
        self.test_accuracy = evaluate_accuracy(self.test_predictions, 
                                            self.test_output)
        self.test_confusion = confusion_matrix(self.number_labels, 
                                            self.test_predictions, 
                                            self.test_output)
    
    @staticmethod
    def prior(output, k):
        ''' Count the empirical proportion of labels of class k among output data.
        
        Parameters
        -----------
        output : numpy.ndarray
            The labels corresponding to some dataset
        k : int 
            The class label we are interested in

        Returns
        --------
        proportion : float
            The fraction of class k observations
        '''

        frequency = np.count_nonzero(output == k)
        proportion = frequency / output.shape[0]

        return proportion
          
    @staticmethod
    def class_covariance(features_subset):
        ''' Calculate a covariance matrix for a mono-labeled feature array.
        
        Parameters
        -----------
        features_subset : numpy.ndarray
            The design matrix of explanatory variables.

        Returns
        --------
        class_cov : numpy.ndarray
            The class-specific covariance matrix for QDA.
        '''

        sample_size = features_subset.shape[0]
        class_mean = np.mean(features_subset, axis=1)
        centered_features_subset = features_subset - class_mean
        unscaled_class_cov = centered_features_subset @ np.T(centered_features_subset)
        if sample_size == 1:
            class_cov = 1/(sample_size) * unscaled_class_cov
        else: class_cov = 1/(sample_size - 1) * unscaled_class_cov

        return class_cov
    
    @staticmethod
    def pooled_covariance(features, output, num_labels):
        ''' Calculate the pooled covariance matrix (used for all classes).
        
        Parameters
        -----------
        features : numpy.ndarray
            The design matrix of explanatory variables.
        output : numpy.ndarray
            The output labels corresponding to features.
        num_labels : numpy.ndarray
            The number of labels present in the data

        Returns
        --------
        pooled_cov : numpy.ndarray
            The pooled covariance matrix for LDA.
        '''

        dimension = features.shape[1]
        sample_size = features.shape[0]
        init_cov = np.zeros((dimension, dimension))

        for k in range(num_labels):
            features_subset = QDA.subset_class(features, output, k)
            sample_size_subset = features_subset.shape[0]
            class_cov = QDA.class_covariance(features_subset, k)
            init_cov += class_cov * (sample_size_subset + 1)
        
        pooled_cov = 1/(sample_size - num_labels) * init_cov

        return pooled_cov

    def discriminant(self, point, k):
        ''' Evaluate the kth quadratic discriminant function at a point.

        Parameters
        -----------
        point : numpy.ndarray
            The point to evaluate at
        k : int
            The class label of interest

        Returns
        --------
        discrim_term : float
            The value of the discriminant function at this point.
        '''

        class_cov = self.class_feature_covs[:, :, k]
        mean_term = self.feature_means[:, k]
        det_term = np.linalg.det(class_cov)
        inv_term = np.linalg.inv(class_cov)
        prior_term = np.log(self.priors[k])

        discrim_term = -0.5*det_term - 0.5*np.T(point - mean_term) @ inv_term @ (point - mean_term) + prior_term

        return discrim_term
  
    def predict_one(self, point):
        '''Predict the label of a test point given a trained model.

        Parameters
        -----------
        point : numpy.ndarray
            The test datapoint we wish to classify.

        Returns
        --------
        label : int
            The predicted class of the point.
        '''

        discrims = np.array([self.discriminant(point, k) for k in range(self.number_labels)], dtype = np.int8)
        label = np.where(dicrims = np.min(discrims))[0]
        return label
    
    def predict_many(self, points):
        '''Predict the label of a matrix of test points given a trained model.

        Parameters
        -----------
        points : numpy.ndarray
            The test datapoints we wish to classify.

        Returns
        --------
        label : int
            The predicted classes of the points.
        '''
        labels = np.apply_along_axis(self.predict_one, 1, points).astype(int8)
        return labels

    
