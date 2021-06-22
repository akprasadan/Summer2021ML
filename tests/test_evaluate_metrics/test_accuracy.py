'''This module tests our accuracy calculator in evaluation_metrics.'''


from src.helperfunctions.evaluation_metrics import evaluate_accuracy
import numpy as np
import pytest


def test_argument_order():
    predictions = np.array([1, 2, 3, 4])
    truth = np.array([4,3,2,1])

    assert evaluate_accuracy(predictions, truth) == evaluate_accuracy(truth, predictions)
    assert evaluate_accuracy(predictions, truth) == 0


def test_comprehensive_error_count():
    '''Some basic checks of accuracy values.'''

    predictions = np.array([1, 2, 3, 4])

    # Just one wrong
    truth = np.array([[1, 2, 3, 4],
                     [0, 2, 3, 4], [1, 0, 3, 4],  # one wrong
                     [0, 0, 3, 4], [1, 2, 0, 0], # two wrong
                     [0, 0, 0, 4], [1, 0, 0, 0], # three wrong
                     [0, 0, 0, 0]])  # 4 wrong

    correct_accuracies = np.array([1, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0])
    for i in range(8):
        assert evaluate_accuracy(predictions, truth[i, :]) ==\
            correct_accuracies[i]


def test_floats_args():
    '''Floats don't alter the accuracy unless 
    inputs are no longer integers.'''

    predictions = np.array([1, 2, 3, 4])
    truth = np.array([[1., 2, 3, 4],  # All correct
                     [1., 2.00000, 3.00, 4.0],  # Still correct
                     [1.2, 2, 3, 4]])  # One wrong
    correct_accuracies = np.array([1, 1, 0.75])
    for i in range(3):
        assert evaluate_accuracy(predictions, truth[i, :]) ==\
            correct_accuracies[i]   


def test_typeerror_exception():
    '''Check that not returning 1D array for each 
    set of labels gives an error.'''

    with pytest.raises(Exception):
        bad_prediction = np.array([[1, 2], [3, 4]])
        good_truth = np.array([1, 2])
        assert evaluate_accuracy(bad_prediction, good_truth)
        assert evaluate_accuracy(good_truth, bad_prediction)
