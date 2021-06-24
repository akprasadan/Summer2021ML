'''This module tests our linear discriminant classifier.'''

from src.classification.lda import LDA
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets

# def test_class_cov():
#     Check our computation of a covariance matrix.

#     n = 10
#     d = 4
#     trials = 5

#     # Compare with numpy_cov
#     for _ in range(trials):
#         features = np.random.rand(n, d)
#         center = np.mean(features, axis = 0)
#         centered_features = features - center
#         aklearn_cov = (n - 1)*QDA.class_covariance(centered_features)
#         numpy_cov = (n - 1)*np.cov(centered_features, rowvar = False)
#         assert np.allclose(numpy_cov, aklearn_cov)

def test_with_sklearn():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    model = LDA(X, y, split_proportion=1)
    lda = LinearDiscriminantAnalysis()
    skmodel = lda.fit(model.train_features, model.train_output)
    skpredictions = skmodel.predict(model.train_features)
    akpredictions = model.train_predictions
    proportion_agree = np.sum(skpredictions == akpredictions) / X.shape[0]
    assert proportion_agree > 0.8

def test_many_sklearn():
    # Using small n causes collinearity/errors.
    n = 50
    d = 4
    num_classes = 3
    trials = 20
    for _ in range(trials):
        X = 5*np.random.rand(n, d)
        y = np.random.randint(0, num_classes, n)
        model = LDA(X, y, split_proportion=1)
        lda = LinearDiscriminantAnalysis()
        skmodel = lda.fit(model.train_features, model.train_output)
        skpredictions = skmodel.predict(model.train_features)
        akpredictions = model.train_predictions
        proportion_agree = np.sum(skpredictions == akpredictions) / n
        assert proportion_agree > 0.8

test_with_sklearn()