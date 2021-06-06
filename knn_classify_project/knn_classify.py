import numpy as np
import pandas as pd
from scipy import stats
import distance_func as metric

class KNNClassify:
    '''
    A class to represent a k-means clustering model.

        Attributes
        ----------
        k : int
            Number of neighbors 
        d : int
            Dimension of dataset (number of features)
        distance_metric : function
            The distance metric to use, imported from distance_func.py. 
            It is the L_2 norm by default (euclidean2), but can also take L_{infinity} (euclidean_infty) and L_1 (euclidean_1).

        train_features : numpy array
            Training data (features only); dimension n_test by d 
        train_labels : numpy array
            Training labels; dimension n_train by 1 
        n_train : int
            Number of training observations

        test_features : numpy array
            Testing data (features only); dimension n_test by d 
        true_test_labels : numpy array
            True testing labels; dimension n_test by 1
        predicted_test_labels : numpy array
            Predicted labels using training data results; dimension n_test bby d
        n_test : int
            Number of testing observations

        test_accuracy : float
            Proportion of correctly classified test labels
        

        Methods
        -------
        k_neighbors_idx : Compute indices of k-nearest neighbors in training data to the given test datapoint

        classify_using_neighbors : Compute most common class among the k-nearest neighbors

        prediction_accuracy : Obtain proportion of correctly classified test labels

        standardize_data_type : Convert data to numpy array if it is a Pandas DataFrame

        fit : Perform the k-nearest neighbors classification algorithm (taking in a Pandas DataFrame or Numpy Array)

    '''
    def __init__(self, k, weight = False, distance_metric = metric.euclidean_2):
        self.k = k 
        self.weight = weight
        self.d = None
        self.train_features = None
        self.train_labels = None
        self.n_train = None
        self.test_features = None
        self.true_test_labels = None
        self.predicted_test_labels = None
        self.n_test = None
        self.test_accuracy = None
        self.distance_metric = distance_metric

    def k_neighbors_idx(self, test_row):
        '''
        Compute the indices in the training data of the k closest neighbors to the test datapoint associated to the given row index.
        
            Parameters:
                test_row : int 
                    Index of observation in test data we wish to classify

            Returns:
                closest_neighbor_idx : numpy array
                    An array of length k, storing the indices of nearest neighbors in training data
        '''

        current_point = self.test_features[test_row, :] # The test observation with row index test_row

        # Calulate distance (Euclidean L_2 norm) of each training observation to current_point
        distances = np.array([self.distance_metric(current_point - self.train_features[i, :]) for i in range(self.n_test)]) 

        closest_neighbor_idx = np.argsort(distances)[:self.k] # Indices (in training data) of closest points

        return closest_neighbor_idx

    def classify_using_neighbors(self, test_row):
        '''
        Compute the predicted classification of the test datapoint with given row index based on training data.
        
            Parameters:
                test_row : int 
                    Index of observation in test data we wish to classify

            Returns:
                classified_label : int
                    Predicted class label
        '''
        closest_neighbor_idx = self.k_neighbors_idx(test_row) # Indices (in training data) of closest points
        neighbor_labels = self.train_labels[closest_neighbor_idx] # Get the labels of closest neighbors
        classified_label = stats.mode(neighbor_labels)[0][0] # Pick the majority class label
        return classified_label

    def prediction_accuracy(self):
        '''Compute the proportion of correctly classified test labels using the results of the KNN algorithm.'''
        
        # Number of correct classifications in test data
        num_matched_labels = np.count_nonzero(self.predicted_test_labels == self.true_test_labels)

        accuracy = num_matched_labels / self.n_test # Proportion correctly classified

        return accuracy
    
    def standardize_data_type(self, data_input):
        ''' Convert data input to numpy array version, if needed.
        '''

        if type(data_input) is np.ndarray:
                return data_input
        elif type(data_input) is pd.DataFrame:
                return data_input.to_numpy()


    def fit(self, train_features, train_labels, test_features, true_test_labels):
        '''
        Train and evaluate the k-nearest neighbor model. Results stored as attributes.
        
            Parameters:
                train_features : numpy array
                    Training data (features only); dimension n_test by d 
                train_labels : numpy array
                    Training labels; dimension n_train by 1 
                test_features : numpy array
                    Testing data (features only); dimension n_test by d 
                true_test_labels : numpy array
                    True labels for test data; dimension n_test by 1
        '''
        self.train_features = self.standardize_data_type(train_features)
        self.train_labels = self.standardize_data_type(train_labels)
        self.test_features = self.standardize_data_type(test_features)
        self.true_test_labels = self.standardize_data_type(true_test_labels)
        self.d = self.train_features.shape[1]
        self.n_train = self.train_features.shape[0]
        self.n_test = self.test_features.shape[0]

        predicted_test_labels = np.array([self.classify_using_neighbors(row) for row in range(self.n_test)])
        self.predicted_test_labels = predicted_test_labels # Vector of predicted test labels
        self.test_accuracy = self.prediction_accuracy() # Test accuracy

model = KNNClassify(k = 3, distance_metric=metric.euclidean_infty)

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'])

model.fit(X_train, y_train, X_test, y_test)

print('n_train =', model.n_train, 'n_test =', model.n_test, 'accuracy =', model.test_accuracy, 'k =', model.k)
