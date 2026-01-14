from src.hw7_3 import t_regression_fit
import numpy as np

# Test t regression design
def test_t_desgin():
    np.random.seed(0)
    n = 200
    X = np.random.randn(n,3)
    beta = np.array([1.0, 2.0, 3.0])
    Y = 1.5 + X @ beta + np.random.standard_t(df = 5, size = n) * 0.1

    result = t_regression_fit(Y,X)
    params = result.params

    # Test for params number colums + intercept + df + sigma
    assert len(params) == X.shape[1] + 1 + 2
    #Test for unnegative attribute of sigma
    assert params[-1] > 0
    # Test for the t distribution make sense
    assert params[-2] > 2
    # Test for validity of regression
    assert not np.any(np.isnan(params))


# sanity test: when our GDP is right -> our estimation should be reasonable
def test_t_regrssion():
    np.random.seed(42)
    n = 500
    X = np.random.randn(n,3)
    beta_true = np.array([2.0,3.0,4.0])
    intercept = 1.0
    Y = intercept + X @ beta_true + np.random.standard_t(df = 8, size = n) * 0.05

    result = t_regression_fit(Y,X)
    params = result.params
    beta_hat = params[1:4]

    assert np.allclose(beta_hat, beta_true, atol = 0.2)

