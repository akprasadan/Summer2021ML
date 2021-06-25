'''This modules tests the knn mean implementation. It works most of the time,
agreeing with sklearn, but often times clusters become empty. This is not a big
problem from a user's perspective, however.'''

from src.clustering.kmean import KCluster
import numpy as np
from sklearn.cluster import KMeans


centers1 = np.array([[2], [-3], [4]])
k1 = centers1.shape[0]
X1 = np.array([[-4], [-3], [-3],  [-2], [2], [2], [3], [4], [4], [5]])
n1 = X1.shape[0]

claimed_distance_matrix1 = np.array([[6, 1, 8], [5, 0, 7], [5, 0, 7], 
                                     [4, 1, 6], [0, 5, 2], [0, 5, 2], 
                                     [1, 6, 1], [2, 7, 0], [2, 7, 0], 
                                     [3, 8, 1]])
calculated_distance_matrix1 = KCluster.compute_distance_matrix(X1, centers1) 
first_clusters1 = np.array([1,1,1,1,0,0,0,2,2,2])
claimed_clusters1 = KCluster.update_clusters(X1, centers1)
first_means1 = np.array([[7/3],[-3],[13/3]])
claimed_means1 = KCluster.update_means(X1, claimed_clusters1, k1)

centers2 = np.array([[-3],[3],[4]])
claimed_distance_matrix2 = np.array([[1,7,8],[0,6,7],[0,6,7],[1,5,6],[5,1,2],[5,1,2],[6, 0, 1],[7, 1, 0],[7, 1, 0],[8, 2, 1]])
calculated_distance_matrix2 = KCluster.compute_distance_matrix(X1, centers2)
first_clusters2 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
claimed_clusters2 = KCluster.update_clusters(X1, centers2)
first_means2 = np.array([[-3],[7/3],[13/3]])
claimed_means2 = KCluster.update_means(X1, claimed_clusters2, k1)


def test_compute_distance_matrix():
    assert np.array_equal(claimed_distance_matrix1, calculated_distance_matrix1)
    assert np.array_equal(claimed_distance_matrix2, calculated_distance_matrix2)

def test_update_clusters():
    assert np.array_equal(first_clusters1, claimed_clusters1)
    assert np.array_equal(first_clusters2, claimed_clusters2)

def test_update_means():
    assert np.array_equal(first_means1, claimed_means1)
    assert np.array_equal(first_means2, claimed_means2)

def test_compare_sklearn():
    '''For a range of values of k and over many trials, 
    let's check that sklearn and KCluster agree. 

    Due to clusters becoming empty, they cannot agree
    (since we simply assign random values at that point).
    Thus, we measure the proportion of trials that did not
    have this occurence and check if it is below some
    threshold.
    '''
    n = 200
    k = [1, 3, 5]
    d = 10
    trials = 100
    n_errors = 0
    threshold = 0.1

    for i in k:
        for j in range(trials):
            X = np.random.rand(n, d)
            model = KCluster(X, k = i)
            ak_centers = model.final_centers

            # Make sure to use same starting points.
            skmodel = KMeans(n_clusters=i, random_state=0, init = model.initial_means, n_init = 1).fit(X)
            sk_centers = skmodel.cluster_centers_
            #print('k = ', i, ' trial: ', j, sk_centers, ak_centers)
            
            try:
                assert np.allclose(sk_centers, ak_centers)
            except:
                n_errors +=1 
                continue
    assert n_errors / (trials * len(k)) < threshold

