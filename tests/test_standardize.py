'''This module checks whether our standardization functionality
behaves as expected. We include a few basic examples (constant or
non-constant columns), along with randomly generated ones that
we compare to sklearn's StandardScalar functionality.
'''

from src.helperfunctions.preprocessing import scale_and_center
import numpy as np
from sklearn.preprocessing import StandardScaler


def test_constant_columns():
    ''' A matrix with each column some scaled version of the all ones vector
        should become all 0.'''
    x1 = np.array([[1, 2], [1, 2]])
    x2 = np.array([[1, 2]])

    assert np.count_nonzero(scale_and_center(x1)) == 0
    assert np.count_nonzero(scale_and_center(x2)) == 0

    # If it is not scaling of all ones, this won't happen
    x3 = np.array([[1, 1],[2, 2]])
    assert np.count_nonzero(scale_and_center(x3)) != 0


def test_compare_sklearn():
    '''Check on randomly generated examples whether sklearn and aklearn agree
    on standardization.'''

    n_cols = 30
    n_rows = 40
    constant = 5
    random_data = constant*np.random.rand(n_rows, n_cols) - constant*np.ones((n_rows, n_cols))
    scaler = StandardScaler()
    scaler.fit(random_data)
    normalized_sklearn = scaler.transform(random_data)
    normalized_aklearn = scale_and_center(random_data)
    difference = np.subtract(normalized_sklearn, normalized_aklearn)
    infty_norm_difference = np.abs(difference).max()

    assert infty_norm_difference < 1e-5