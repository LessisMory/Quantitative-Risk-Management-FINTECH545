import pandas as pd
import numpy as np
info = pd.read_csv("test2.csv")
n = info.shape[0]
lam = 0.97

def ew_cov(info, n, lam):
    data = info.copy()
    index = np.arange(n)
    weight = (1 - lam) * lam ** (n - index - 1)
    cum_weight = weight.sum()
    weight = weight / cum_weight
    mean = np.sum(weight[:,None] * data, axis = 0)
    data = data - mean
    Cov = data.T @ (weight[:,None] * data)
    return Cov

Cov = ew_cov(info, n, lam)
Cov.to_csv("myoutput2.1.csv", index=False)