import numpy as np
import pandas as pd

info = pd.read_csv("test2.csv")
n = info.shape[0]

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

Cov_97 = ew_cov(info, n, 0.97)
Cov_94 = ew_cov(info, n, 0.94)
SD_97 = np.diag(np.sqrt(np.diag(Cov_97)))
InvSD_94 = np.diag(1.0 / np.sqrt(np.diag(Cov_94)))
Cov_target = SD_97 @ InvSD_94 @ Cov_94 @ InvSD_94 @ SD_97
Cov_target.to_csv("myoutput2.3.csv", index=False)