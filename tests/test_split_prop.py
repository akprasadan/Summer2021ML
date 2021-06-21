from src.helperfunctions.preprocessing import train_test_split
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
        split = train_test_split(X, y, prop) # A namedtuple object

        assert split.sample_size == n
        assert split.train_size == desired_trains[i]
        assert split.test_size == 40 - desired_trains[i]

        # Make sure the train and test splits do not intersect
        assert np.unique(np.append(split.train_rows, split.test_rows)).shape[0] == n
    


