import knn_classify as knn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()


# Let's verify that our preprocessing was valid
# We compare it to sklearn's Standard Scalar function
def test_normalize():
    n_cols = 30
    n_rows = 40
    constant = 5
    random_data = constant*np.random.rand(n_rows, n_cols) -  constant*np.ones((n_rows, n_cols))
    scaler = StandardScaler()
    scaler.fit(random_data)
    normalized_sklearn = scaler.transform(random_data)
    normalized_mine = knn.KNNClassify.scale_and_center(random_data)
    difference = np.subtract(normalized_sklearn, normalized_mine)
    assert np.abs(difference).max() < 1e-14
    #return np.testing.assert_allclose(difference, np.zeros((n_rows, n_cols)))
