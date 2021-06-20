from src.regression.linearreg import Linear
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def linear_reg_standardize(X, y, decision):
    aklearn_linear = Linear(X, y, split_proportion=1, standardized=decision)

    if decision:
        normalized_X = StandardScaler().fit(X).transform(X)
        sklearn_linear = LinearRegression().fit(normalized_X, y)

    else:
        sklearn_linear = LinearRegression().fit(X, y)

    sklearn_coeff = np.append(sklearn_linear.intercept_, sklearn_linear.coef_)
    difference = np.subtract(aklearn_linear.coefficients, sklearn_coeff)
    infty_norm_difference = np.abs(difference).max()

    return infty_norm_difference


def test_coeff_std_sklearn():
    '''Compare sklearn's coefficient fit to aklearn's;
    we do so for both standardized/non-standardized data.

    Importantly, we make sure the feature matrix does not
    become singular by standardizing the all ones column,
    as otherwise the matrix would be singular and aklearn
    would run into an error.'''

    X, y = load_boston(return_X_y=True)
    error_standardized = linear_reg_standardize(X, y, True)
    error_not_standardized = linear_reg_standardize(X, y, False)

    assert error_standardized < 1e-5
    assert error_not_standardized < 1e-5
