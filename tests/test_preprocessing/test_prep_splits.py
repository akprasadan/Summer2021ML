from src.helperfunctions.preprocessing import train_test_split
from src.helperfunctions.preprocessing import cross_validation_folds_idx

import numpy as np


def test_train_split_prep():
    '''Check that we are splitting correctly.'''

    n = 40
    dimension = 5
    X = np.random.rand(n, dimension)
    y = np.random.rand(n, 1)

    # Proportions of training data
    split_prop = [0, 0.25, 0.5, 0.75, 1]
    desired_trains = [0, 10, 20, 30, 40]

    for i, prop in enumerate(split_prop):
        split = train_test_split(X, y, prop)  # A namedtuple object

        assert split.sample_size == n
        assert split.train_size == desired_trains[i]
        assert split.test_size == 40 - desired_trains[i]

        # Make sure the train and test splits do not intersect
        assert np.unique(np.append(split.train_rows, split.test_rows)).\
            shape[0] == n


def test_cross_val():
    row_count = 21
    fold_count = 4

    folds = cross_validation_folds_idx(row_count, fold_count)

    assert folds.shape[0] == 4  # Fold count
    assert folds.shape[1] == 5  # 5 indices per row/fold

    # All but exactly one row is included
    assert len(np.unique(folds)) == 20  

    #  Check that every entry of folds is an integer
    assert np.all(np.equal(np.mod(folds, 1), 0))  
