'''This module tests the Logistic class and its methods.
We compare the results to sklearn, and also verify that
we are correctly calculating the sigmoid function.'''


from src.classification.logisticreg import Logistic
import numpy as np
from sklearn import linear_model
import pytest 


def test_probability_estimate1():
    '''Check that we compute the sigmoid correctly for one input.
    Let's try null inputs.'''
    
    # Check the null coefficient
    row_1 = np.random.rand(10)
    coeff_1 = np.zeros(10)
    result_1 = Logistic.probability_estimate(row_1, coeff_1)
    assert result_1 == 0.5

    row_2 = np.zeros(10)
    coeff_2 = np.random.rand(10)
    result_2 = Logistic.probability_estimate(row_2, coeff_2)

    assert result_2 == 0.5


def test_probability_estimate2():
    '''Test we compute dot product in each component correctly.'''
    row = np.array([1, 2, 3, 4])
    coeff_1 = np.array([1, 0, 0, 0])
    coeff_2 = np.array([0, 1, 0, 0])
    coeff_3 = np.array([0, 0, 1, 0])
    coeff_4 = np.array([0, 0, 0, 1])
    
    result_1 = Logistic.probability_estimate(row, coeff_1)
    result_2 = Logistic.probability_estimate(row, coeff_2)
    result_3 = Logistic.probability_estimate(row, coeff_3)
    result_4 = Logistic.probability_estimate(row, coeff_4)

    assert result_1 == 1/(1+np.exp(-1))
    assert result_2 == 1/(1+np.exp(-2))
    assert result_3 == 1/(1+np.exp(-3))
    assert result_4 == 1/(1+np.exp(-4))

    
def test_probability_estimate3():
    '''Test sigmoid for multiple rows at once.
    Check that coefficients cannot have this property.'''
    row_1 = np.array([1, 2, 3, 4])
    row_2 = np.zeros(4)
    feature_matrix = np.vstack((row_1, row_2))
    coeff = np.array([1, 2, 3, 4])

    claimed_result = np.array([1/(1+np.exp(-30)), 1/2])
    result = Logistic.probability_estimate(feature_matrix, coeff)
    assert np.allclose(result, claimed_result)

    with pytest.raises(Exception):
        #  Coefficient is insufficiently sized, feature matrix is fine
        coeff = np.array([1, 2, 3])
        assert Logistic.probability_estimate(feature_matrix, coeff)

        coeff = np.array([[1,2,3,4], [5,6,7,8]])

        # One good row, one bad 2D coefficient
        assert Logistic.probability_estimate(row_1, coeff)
        # Two good rows, one bad 2D coefficient
        assert Logistic.probability_estimate(feature_matrix, coeff)

def test_coefficients():
    '''Test how the logistic regression coefficients compare to sklearn.
    It turns out that we quickly (< 10 steps) hit a difference of 10^(-5) between
    consecutive coefficient estimates, but improving beyond 
    10^(-6) is very hard.'''

    n = 200
    d = 10
    num_trials = 5

    for _ in range(num_trials):
        X = 5*np.random.rand(n, d)
        y = np.random.randint(0, 2, n)
        model = Logistic(X,y, standardized = True, split_proportion=1, 
                         tolerance=1e-32, max_steps = 100)
        aklearn_results = model.coefficients

        # Note that sklearn penalizes by default
        clf = linear_model.LogisticRegression(penalty = 'none', 
                                              max_iter = 1000, tol=1e-15)
        clf.fit(X, y)
        sklearn_results = np.append(clf.intercept_, clf.coef_)
        
        difference = np.linalg.norm(sklearn_results - aklearn_results)
        assert difference < 1e-4

def test_weight_matrix():
    probabilities = np.array([0, 1, 0.25, 0.75, 0.3])
    result = Logistic.weighted_matrix(probabilities)
    truth = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 3/16, 0, 0],
                     [0, 0, 0, 3/16, 0],
                     [0, 0, 0, 0, 0.21]])

    assert np.allclose(truth, result)
