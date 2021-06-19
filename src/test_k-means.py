import k_means as km
import numpy as np

example1 = km.KCluster()
centers1 = np.array([[2],[-3],[4]])
k1 = centers1.shape[0]
X1 = np.array([[-4],[-3],[-3], [-2],[2],[2],[3],[4],[4],[5]])
n1 = X1.shape[0]
claimed_distance_matrix1 = np.array([[6,1,8],[5,0,7],[5,0,7],[4,1,6],[0,5,2],[0,5,2],[1,6,1],[2,7,0],[2,7,0],[3,8,1]])
calculated_distance_matrix1 = example1.compute_distance_matrix(X1, centers1, n1, k1) 
first_clusters1 = np.array([1,1,1,1,0,0,0,2,2,2])
claimed_clusters1 = example1.update_clusters(X1, centers1, n1, k1)
first_means1 = np.array([[7/3],[-3],[13/3]])
claimed_means1 = example1.update_means(X1, claimed_clusters1, k1)

example2 = km.KCluster()
centers2 = np.array([[-3],[3],[4]])
claimed_distance_matrix2 = np.array([[1,7,8],[0,6,7],[0,6,7],[1,5,6],[5,1,2],[5,1,2],[6, 0, 1],[7, 1, 0],[7, 1, 0],[8, 2, 1]])
calculated_distance_matrix2 = example2.compute_distance_matrix(X1, centers2, n1, k1) 
first_clusters2 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
claimed_clusters2 = example2.update_clusters(X1, centers2, n1, k1)
first_means2 = np.array([[-3],[7/3],[13/3]])
claimed_means2 = example2.update_means(X1, claimed_clusters2, k1)


def test_compute_distance_matrix():
    assert np.array_equal(claimed_distance_matrix1, calculated_distance_matrix1)
    assert np.array_equal(claimed_distance_matrix2, calculated_distance_matrix2)

def test_update_clusters():
    assert np.array_equal(first_clusters1, claimed_clusters1)
    assert np.array_equal(first_clusters2, claimed_clusters2)

def test_update_means():
    assert np.array_equal(first_means1, claimed_means1)
    assert np.array_equal(first_means2, claimed_means2)

def test_predict_method():
    for _ in range(0, 5):
        X = np.random.rand(15,6)
        model = km.KCluster()
        model.fit(X)
        model.predict(X)
        assert np.array_equal(model.predicted_means, model.training_data_means)

