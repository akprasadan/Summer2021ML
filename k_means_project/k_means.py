import numpy as np 
from sklearn.utils.random import sample_without_replacement

class Cluster:
    def __init__(self):
        self.training_data = None
        self.testing_data = None
        self.d = None
        self.n_train = None
        self.k = None
        self.clusters_indices = None
        self.means = None
        self.training_data_means = None
        self.errors = None
        self.number_iterations = None
        self.threshold = None
        self.predicted_clusters = None
        self.predicted_means = None
        self.labelled_training_data = None
        self.labelled_testing_data = None

    def compute_distance_matrix(self, X, center_points, n, k):
        '''
        Compute distances of each of n row vectors in a matrix to each of k vectors representing 'centers.'

            Parameters:
                X (numpy array): n by d matrix, where d is the dimension of dataset, n is the sample size
                center_points (numpy array): k by d matrix, where k is the number of clusters
                n (int): sample size of dataset
                k (int): number of clusters

            Returns:
                distance_matrix (numpy array): 
                    An n by k matrix.
                    The (i,j) entry is the distance from the i-th row of X to the jth row of centers.  
                    We use the l_2 Euclidean norm for d dimension.              

        '''
        distance_matrix = np.zeros((n, k)) # Initialize the matrix

        for row in range(n):
            for column in range(k):
                distance_matrix[row, column] = np.linalg.norm(center_points[column, :] - X[row, :])
        
        return distance_matrix
    
    def update_clusters(self, X, center_points, n, k):
        '''
        Given a current set of k cluster points, compute which cluster each row vector of the data matrix is closest to.
        
            Parameters:
                X (numpy array): n by d matrix, where d is the dimension of dataset, n is the sample size
                center_points (numpy array): k by d matrix, where k is the number of clusters
                n (int): sample size of dataset
                k (int): number of clusters

            Returns:
                closest_center_indices (numpy array): 
                    A vector of length n.
                    The i-th entry is the integer p minimizing distance of ith row of X to the pth cluster point.
                    We use the l_2 Euclidean norm for d dimension.   
        '''
        distance_matrix = self.compute_distance_matrix(X, center_points, n, k)
        
        # For each row i, we want to know the column k for which distance_matrix[i,k] is minimal
        # We will store this in closest_center_indices

        closest_center_indices = np.zeros(n)
        for row in range(n):
            distance_to_centers = distance_matrix[row, :]
            closest_center_index = np.where(distance_to_centers == np.amin(distance_to_centers))[0][0] 
            closest_center_indices[row] = closest_center_index
        return closest_center_indices.astype(int)

    def update_means(self, X, closest_center_indices, k):
        '''
        Compute the mean of each current cluster.
        
            Parameters:
                X (numpy array): n by d matrix, where d is the dimension of dataset, n is the sample size
                closest_center_indices (numpy array): 
                    A vector of length n.
                    The i-th entry is the integer p minimizing distance of ith row of X to the pth cluster point.
                k (int): number of clusters

            Returns:
                cluster_means (numpy array): 
                    A k by d matrix.
                    The i-th row is the d-dimensional element-wise mean of all row vectors of X in cluster i.
        '''

        d = X.shape[1]
        cluster_means = np.zeros((k, d))

        for cluster_index in range(k):
            cluster_subset = np.where(closest_center_indices == cluster_index)[0]
            cluster_means[cluster_index, :] = np.mean(X[cluster_subset, :], axis = 0)

        return cluster_means

    def fit(self, X, k = 3, threshold = 0.01):
        '''
        Fit k-means to a dataset up to a predetermined threshold of convergence. Results stored as attributes.
        
            Parameters:
                X (numpy array or Pandas DataFrame): n by d matrix, where d is the dimension of dataset, n is the sample size
                k (int): number of clusters
                threshold (float): 
                    Positive number that the error must fall below for the algorithm to terminate.
                    Error is defined as Frobenius norm of difference of the matrix of cluster means between updates.

                
        '''
        self.training_data = X
        self.n_train = self.training_data.shape[0]
        self.d = self.training_data.shape[1]
        self.k = k
        self.threshold = threshold

        # Pick which rows of our data we will initialize as clusters (uniformly random)
        initial_data_subset = sample_without_replacement(n_population = self.n_train, n_samples = self.k, random_state = 100)
        initial_data_subset.sort()

        # Store these row vectors as our first set of means, comprising the first cluster
        # An k times d matrix
        initial_means = self.training_data[initial_data_subset, :] 

        # Initialize arbitrary large error, so first step will always run
        error = np.inf
        errors = [] # Store the error at each step (see docstring)
        total_steps = 0 # How many iterations get performed?

        while error > threshold:
            total_steps += 1
            updated_indices = self.update_clusters(self.training_data, initial_means, self.n_train, self.k)
            updated_means = self.update_means(self.training_data, updated_indices, self.k)
            error = np.linalg.norm(initial_means - updated_means) # How much did the k-means move?
            errors.append(error)
            initial_means = updated_means # Make the update and start the enxt round

        self.clusters_indices = updated_indices
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
        predicted_clusters = self.update_clusters(self.testing_data, self.means, n_predictions, self.k)
        self.predicted_clusters = predicted_clusters

        predicted_means = np.array([self.means[predicted_clusters[i], :] for i in range(n_predictions)])
        # The ith prediction is the jth row of the means, where j = predicted_clusters[i]

        self.predicted_means = predicted_means
        self.labelled_testing_data = np.append(self.testing_data, self.predicted_clusters.reshape((self.n_train, 1)), axis = 1)



example = Cluster()

X = np.random.rand(20,5)
example.fit(X)
print(example.labelled_training_data)
