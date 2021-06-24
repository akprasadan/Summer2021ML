'''This module tests the confusion matrix generator in evaluation_metrics.'''

from src.helperfunctions.evaluation_metrics import confusion_matrix
import numpy as np
from sklearn import metrics

def test_half_wrong():
    '''In this example, we always predict 0.

    We are wrong half the time on the whole, but always for 1,
    and never for 0. Recall that if there are L labels, 
    then for 0 <= i, j <= L - 1, the (i, j) entry 
    contains the number of times we predicted j when the true class is i.'''

    predictions = [0, 0, 0, 0, 0, 0, 0, 0]
    truth = [1, 1, 1, 1, 0, 0, 0, 0]
    conf_matrix = confusion_matrix(2, predictions, truth)

    claim = np.array([[4, 0],  # True = 0, predict = 0; true = 0, predict = 1
                     [4, 0]])  # True = 1, predict = 0; true = 1, predict = 1

    assert np.array_equal(claim, conf_matrix)


def test_always_wrong():
    '''In this example, we always predict 0, the truth is always 1.'''

    predictions = [0, 0, 0, 0, 0, 0, 0, 0]
    truth = [1, 1, 1, 1, 1, 1, 1, 1]
    conf_matrix = confusion_matrix(2, predictions, truth)

    claim = np.array([[0, 0],  # True = 0, predict = 0; true = 0, predict = 1
                     [8, 0]])  # True = 1, predict = 0; true = 1, predict = 1
                     
    assert np.array_equal(claim, conf_matrix)


def test_never_wrong():
    '''In this example, we are always right.'''

    predictions = [1, 1, 1, 1, 0, 0, 0, 0]
    truth = [1, 1, 1, 1, 0, 0, 0, 0]
    conf_matrix = confusion_matrix(2, predictions, truth)

    claim = np.array([[4, 0],  # True = 0, predict = 0; true = 0, predict = 1
                     [0, 4]])  # True = 1, predict = 0; true = 1, predict = 1

    ''' Let's add a second label. 
    Then we embed the above matrix in the top right of a 3 by 3 matrix
    There will be a 1 in the (3,3) position, and 0s in the other new entries.
    Lastly, since we removed a predict 1/true 1, we reduce the (1,1) entry by 1.'''

    predictions = [2, 1, 1, 1, 0, 0, 0, 0]
    truth = [2, 1, 1, 1, 0, 0, 0, 0]
    conf_matrix = confusion_matrix(3, predictions, truth)

    claim = np.array([[4, 0, 0], 
                     [0, 3, 0],
                     [0, 0, 1]])  

    assert np.array_equal(claim, conf_matrix)

    predictions = [2, 1, 1, 1, 2, 0, 0, 0]
    truth =       [2, 1, 1, 1, 1, 0, 0, 1]
    conf_matrix = confusion_matrix(3, predictions, truth)

    claim = np.array([[2, 0, 0], 
                     [1, 3, 1],
                     [0, 0, 1]])  

    assert np.array_equal(claim, conf_matrix)


def test_sum():
    '''The elements of the matrix should sum to the test sample size.

    We observe n combinations of prediction and truth.'''
    loop_count = 10
    num_labels = np.random.randint(2, 10, loop_count)
    sample_sizes = np.random.randint(100, 200, loop_count)
    for i in range(loop_count):
        n = sample_sizes[i]
        num_label = num_labels[i]
        predictions = np.random.randint(0, num_label, n) # randint(0,n) excludes n
        truth = np.random.randint(0, num_label, n)
        conf_matrix = confusion_matrix(num_label, predictions, truth)
        assert np.sum(conf_matrix) == n

def test_with_sklearn():
    '''Ensure that sklearn and aklearn have same 
    confusion matrix output.'''

    loop_count = 10
    num_labels = np.random.randint(2, 10, loop_count)
    sample_sizes = np.random.randint(100, 200, loop_count)
    for i in range(loop_count):
        n = sample_sizes[i]
        num_label = num_labels[i]
        predictions = np.random.randint(0, num_label, n) # randint(0,n) excludes n
        truth = np.random.randint(0, num_label, n)
        conf_matrix = confusion_matrix(num_label, predictions, truth)
        sklearn_conf = metrics.confusion_matrix(truth, predictions)
        assert np.array_equal(conf_matrix, sklearn_conf)
