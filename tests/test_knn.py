from src.classification.knn_classify import KNNClassify
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def test_knn_sklearn():
    '''Compare knn predictions to sklearn.'''
    n = 100
    d = 5
    neighbor_counts = [1, 3, 5, 7]
    trials = 5

    for k in neighbor_counts:
        for _ in range(trials):
            X = np.random.rand(n, d)
            y = np.random.randint(0, d, n)
            model = KNNClassify(X, y, standardized=False, k = k)

            train_features = model.train_features
            test_features = model.test_features
            train_output = model.train_output
            test_output = model.test_output 
                
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(train_features, train_output)

            assert np.allclose(neigh.predict(test_features), model.predict_class())

