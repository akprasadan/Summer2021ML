from src.classification.qda import QDA
import numpy as np

def test_prior():
    output = np.array([5, 4, 3, 2, 1, 3, 4, 5, 1, 4])
    assert QDA.prior(output, 3) == 2/len(output)
    assert QDA.prior(output, 5) == 2/len(output)
    assert QDA.prior(output, 4) == 3/len(output)
    assert QDA.prior(output, 2) == 1/len(output)