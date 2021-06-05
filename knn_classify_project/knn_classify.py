from os import close
import numpy as np
from scipy import stats


class NeighborFinder:
    def __init__(self):
        self.k = None
        self.d = None

        self.train_features = None
        self.train_labels = None
        self.n_train = None

        self.test_features = None
        self.true_test_labels = None
        self.predicted_test_labels = None
        self.n_test = None

        self.test_accuracy = None

    def all_neighbor_finder(self, test_row):
        current_point = self.test_features[test_row, :] # 1 row
        distances = np.array([np.linalg.norm(current_point - self.test_features[i, :]) for i in range(self.n_train)]) # Length n
        closest_neighbor_idx = np.argpartition(distances, self.k) # Length k

        return closest_neighbor_idx

    def classify_using_neighbors(self, test_row):
        closest_neighbor_idx = self.all_neighbor_finder(test_row)
        neighbor_labels = self.train_labels[closest_neighbor_idx]
        classified_label = stats.mode(neighbor_labels)
        
        return classified_label

    def prediction_accuracy(self):
        num_matched_labels = np.sum(self.predicted_test_labels = self.true_test_labels)
        accuracy = num_matched_labels / self.n_test
        return accuracy

    def fit(self, train_features, train_labels, test_features, k):
        self.train_features = np.array(train_features)
        self.train_labels = np.array(train_labels)
        self.k = k
        self.d = self.train_features.shape[1]
        self.n_train = self.train_features.shape[0]
        self.n_test = self.test_feature.shape[0]

        # Let's compute each test_label
        predicted_test_labels = np.array([self.classify_using_neighbors(row) for row in range(0, self.n_test)])
        self.predicted_test_labels = predicted_test_labels
        self.test_accuracy = self.prediction_accuracy()



