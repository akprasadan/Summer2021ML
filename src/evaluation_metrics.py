'''This module provides functionality to evaluate the 
performance of a machine learning model.
Each method compares the predicted data to true data.

'''

import numpy as np
from numba import jit
import norms as norms


def evaluate_accuracy(predicted_output, true_output):
    """Calculate the proportion of labels a classifier correctly predicts.

    Parameters
    ----------
    predicted_output : numpy.ndarray
        The predictions made by the classifier.
    true_output : int
        The true labels.

    Returns
    -------
    accuracy : float
        Proportion correctly predicted, between 0 and 1 (inclusive).
    """
    number_predictions = predicted_output.shape[0]
    correct_predictions = np.count_nonzero(predicted_output == true_output)
    accuracy = correct_predictions / number_predictions
    return accuracy


def confusion_matrix(number_labels, predicted_output, true_output):
    """Calculate contingency table of predicted labels and true labels.

    If there are L labels, then for 1 <= i, j <= L, the (i, j) entry 
    contains the number of times we predicted j when the true class is i.

    Parameters
    ----------
    number_labels : int
        The number of possible labels in the data.
    predicted_output : numpy.ndarray
        The predictions made by the classifier.
    true_output : int
        The true labels .

    Returns
    -------
    confusion_matrix : numpy.ndarray
        Square [confusion] matrix of size number_labels by number_labels.
    
    Notes
    ------
    The true output and predicted output row vectors are stacked over 
    each other. This 2 by sample_size numpy array is stored as output_combined.
    To identify how often we predict j when the truth is i, we count 
    the number of times the column vector [i, j] appears in output_combined.
    I consulted [1]_ to figure out the lattermost step.

    References
    -----------
    .. [1] https://stackoverflow.com/a/40382459
    """
    confusion_matrix = np.zeros(shape=(number_labels, number_labels), dtype = np.int8)
    print(true_output, predicted_output)
    output_combined = np.stack((true_output, predicted_output), axis=1)  

    for row_index in range(number_labels):  #
        for col_index in range(number_labels):  # j
            count = (output_combined == (row_index, col_index)).\
                all(axis=1).sum()
            confusion_matrix[row_index, col_index] = count

    return confusion_matrix


def evaluate_regression_error(predicted_output, true_output, 
                              norm=norms.euclidean_2):
    """Calculate the error with respect to a norm of regression output.

    Parameters
    ----------
    predicted_output : numpy.ndarray
        The predictions made by the classifier.
    true_output : int
        The true response values.
    norm : func
        The choice of norm to use to measure error. 
        Default is the Euclidean L_2 norm.

    Returns
    -------
    error : float
        Measurement of error of regression model.
    """

    error = norm(predicted_output - true_output)**2
    return error

    