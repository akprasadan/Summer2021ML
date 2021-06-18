import numpy as np 
from sklearn.utils.random import sample_without_replacement
from clustering import Clustering

class KCluster(Clustering):
    '''
    A class to represent a k-means clustering model.

        Attributes
        ----------
        k : int
            How many clusters to use in the algorithm
        clusters_indices : list
            A list of length n storing which cluster each observation belongs to
        means : numpy array
            A k by d matrix storing the positions of the cluster centers
        number_iterations : int
            How many steps the algorithm took to converge
        threshold : float
            How small the error needs to be to end the convergence
        predicted_clusters : list
            A list of length 
        predicted_means : numpy array
            A matrix of size n_predictions by d
        n_predictions : int
            Number of rows in the test set
        labelled_training_data : numpy array
            The training matrix with an additional column of cluster labels
        labelled_testing_data : numpy array
            The testing matrix with an additional column of cluster labels

        Methods
        --------
        compute_distance_matrix : Computes matrix of distances between each observation and each cluster

        update_clusters : Compute list of closest clusters given a distance matrix

        update_means : Obtain matrix of updated cluster centers

        fit : Perform the k-means clustering algorithm (taking in a Pandas DataFrame or Numpy Array)

        predict : Calculate closest clusters to new data using clusters calculated from fit 

    '''

    def __init__(self, features, standardized=True, k=3, threshold=0.01):
        super().__init__(features, standardized)
        self.k = None
        self.clusters_indices = None
        self.means = None
        self.number_iterations = None
        self.threshold = threshold
        self.predicted_clusters = None
        self.predicted_means = None
        self.n_predictions = None
        self.labelled_training_data = None
        self.labelled_testing_data = None

    @staticmethod # Note, REUSE THIS FOR KNN!!
    def compute_distance_matrix(features, centers):
        '''
        Compute distances of each of n row vectors in a matrix to each of k vectors representing 'centers.'

        Parameters
        -----------
        features : numpy.ndarray
            n by d array, where d is the dimension of dataset, n is the sample size
        centers : numpy.ndarray
            k by d array, where k is the number of clusters

        Returns
        --------
        distance_matrix : numpy.ndarray
            An n by k array.
            The (i,j) entry is the distance from the i-th row of X to the jth row of centers.  
            We use the l_2 Euclidean norm for d dimensions.

        Notes
        ------
        An efficient numpy procedure (using its broadcasting functionality) to compute 
        all pairwise differences between two collections of data points is given in [4]_.
        We use this, as an alternative to using a manual nested 'for-loop' procedure.

        References
        ------------
        .. [4] https://sparrow.dev/pairwise-distance-in-numpy/              

        '''

        pairwise_differences = features[:, None, :] - centers[None, :, :]
        distance_matrix = np.linalg.norm(pairwise_differences)
        
        return distance_matrix
    
    @staticmethod
    def update_clusters(features, centers):
        '''
        Given a current set of k centers, compute which cluster each row vector of the data matrix is closest to.
        
        Parameters
        -----------
        features : numpy.ndarray
            n by d array, where d is the dimension of dataset, n is the sample size
        centers : numpy.ndarray
            k by d array, where k is the number of centers

        Returns
        --------
        closest_center_idx : numpy.ndarray 
            A vector of length n whose i-th entry is the integer p minimizing 
            distance of ith row of features to the pth cluster point. 
            We use the d-dimensional l_2 Euclidean norm.
        '''

        distance_matrix = KCluster.compute_distance_matrix(features, centers)
        n = features.shape[0]        

        closest_center_idx = np.zeros(n, dtype = np.int8)

        for row in range(n):
            distance_to_centers = distance_matrix[row, :]
            closest_center_index = np.where(distance_to_centers == np.amin(distance_to_centers))[0][0] 
            closest_center_idx[row] = closest_center_index

        return closest_center_idx.astype(int)

    @staticmethod
    def update_means(features, closest_center_idx, k):
        '''
        Parameters
        -----------
        features : numpy.ndarray
            n by d array, where d is the dimension of dataset, n is the sample size
        centers : numpy.ndarray
            k by d array, where k is the number of centers.
        k : int
            The number of centers.

        Returns
        --------
        cluster_means : numpy.ndarray 
            A k by d array.
            The i-th row is the d-dimensional element-wise mean of all row vectors of X in cluster i.
        '''

        d = features.shape[1]
        cluster_means = np.zeros((k, d))

        for cluster_index in range(k):
            cluster_subset = np.where(closest_center_idx == cluster_index)[0]
            cluster_means[cluster_index, :] = np.mean(features[cluster_subset, :], axis = 0)

        return cluster_means

    def fit(self):
        '''
        Fit k-means to a dataset up to a predetermined threshold of convergence. 
        '''

        # Pick which rows of our data we will initialize as clusters (uniformly random)
        initial_data_idx = sample_without_replacement(n_population = self.sample_size, 
                                                      n_samples = self.k, random_state = 100)
        initial_data_idx.sort()

        # Store these row vectors as our first set of means, comprising the first cluster
        # An k times d array
        initial_means = self.features[initial_data_idx, :] 

        # Initialize arbitrary large error, so first step will always run
        error = np.inf
        errors = [] # Store the error at each step (see docstring)
        total_steps = 0 # How many iterations get performed?

        while error > self.threshold:
            total_steps += 1
            updated_indices = self.update_clusters(self.features, initial_means)
            updated_means = self.update_means(self.features, updated_indices, self.k)
            error = np.linalg.norm(initial_means - updated_means) # How much did the k-means move?
            errors.append(error)
            initial_means = updated_means # Make the update and start the enxt round

        self.final_clusters_idx = updated_indices
        self.means = updated_means
        self.errors = errors
        self.number_iterations = total_steps

        training_means = np.zeros((self.n_train, self.d))

        for i in range(self.n_train):
            # The ith prediction is the jth row of the means, where j = predicted_clusters[i]
            training_means[i] = self.means[updated_indices[i], :]

        self.training_data_means = training_means  
        
        self.labelled_training_data = np.append(self.training_data, self.clusters_indices.reshape((self.n_train, 1)), axis = 1)
  

    def predict(self, testing_data):
        '''
        Predict clusters and their means for test data, stored as attributes. 
            
            Parameters:
                testing_data (numpy array): n_predictions by self.d matrix
  
        '''
        
        self.testing_data = testing_data
        n_predictions = testing_data.shape[0]
        self.n_predictions = n_predictions
        predicted_clusters = self.update_clusters(self.testing_data, self.means, n_predictions, self.k)
        self.predicted_clusters = predicted_clusters

        predicted_means = np.array([self.means[predicted_clusters[i], :] for i in range(n_predictions)])
        # The ith prediction is the jth row of the means, where j = predicted_clusters[i]

        self.predicted_means = predicted_means
        self.labelled_testing_data = np.append(self.testing_data, self.predicted_clusters.reshape((self.n_train, 1)), axis = 1)

