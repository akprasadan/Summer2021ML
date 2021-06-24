'''This module builds a class for k-nearest neighbor classification.
'''

import numpy as np
from numpy.lib.shape_base import split
from src.classification.classification import Classification
from src.helperfunctions.evaluation_metrics import evaluate_accuracy, confusion_matrix
from src.classification.qda import QDA

class LDA(QDA):
    '''
    A class used to represent a linear discriminant analysis classifier.
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

    covariance_matrix : numpy.ndarray
        The pooled covariance matrix used in the discriminant function
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
    qda.QDA : Use the more flexible quadratic discriminant analysis
    '''

    def __init__(self, features, output, split_proportion=0.75,
                 number_labels=None, standardized=True):
        super().__init__(features, output, split_proportion=0.75,
                 number_labels=None, standardized=True)
        self.covariance_matrix = LDA.pooled_covariance(self.train_features,
                                                       self.train_output,
                                                       self.number_labels)
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
            features_subset = features[output == k]
            sample_size_subset = features_subset.shape[0]
            class_cov = QDA.class_covariance(features_subset)
            init_cov += class_cov * (sample_size_subset - 1)
        
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
        discrim : float
            The value of the discriminant function at this point.
        '''

        feature_subset = self.train_features[self.train_output == k]
        mean_term = np.mean(feature_subset, axis = 0)

        inv_term = np.linalg.inv(self.covariance_matrix)
        prior_term = np.log(QDA.prior(self.train_output, k))
        

        discrim = prior_term + point.T @ inv_term @ mean_term - \
            0.5 * mean_term.T @ inv_term @ mean_term
        return discrim
  
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

        discrims = np.array([self.discriminant(point, k) for k in range(self.number_labels)])
        label = np.where(discrims == np.max(discrims))[0][0]
        return label

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target
model = LDA(X, y, split_proportion=1)
model2 = QDA(X, y, split_proportion=1)

