'''
Provides metrics by which we can evaluate the performance of our model.
'''

import numpy as np
from numba import jit
import norms as norms

def evaluate_accuracy(predicted_output, true_output):
    number_predictions = predicted_output.shape[0]
    correct_predictions = np.count_nonzero(predicted_output == true_output)
    return correct_predictions / number_predictions

@jit(nopython=True) 
def confusion_matrix(number_labels, predicted_output, true_output):
    confusion_matrix = np.zeros(shape = (number_labels, number_labels))
    output_combined = np.stack((true_output, predicted_output), axis = 1)  # each row has 2 elements, of form (truth, prediction)

    for row_index in range(number_labels): #i
        for col_index in range(number_labels): #j
            # Let's compute the number of rows of output_combined containing (i,j) 
            # want predicted == j, true == i.
            # Source of the following line's logic: https://stackoverflow.com/a/40382459
            confusion_matrix[row_index, col_index] = (output_combined == (row_index, col_index)).all(axis = 1).sum()

    return confusion_matrix

def evaluate_regression_error(predicted_output, true_output, norm = norms.euclidean_2):
    return norm(predicted_output - true_output)**2