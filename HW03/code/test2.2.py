import pandas as pd 
import numpy as np 

info = pd.read_csv("test2.csv")
n = info.shape[0]
lam = 0.94

def ew_cor(info, n, lam):
    data = info.copy()
    index = np.arange(n)
    lam = 0.94
    weight = (1 - lam) * lam ** (n - index - 1)
    sum_weight = np.sum(weight)
    weight = weight / sum_weight
    mean = np.sum(data * weight[:, None], axis = 0)
    data = data - mean
    Cov = data.T @ (weight[:, None] * data)
    std_vec = np.sqrt(np.diag(Cov))
    Cor = Cov / np.outer(std_vec, std_vec)
    return Cor

Cor = ew_cor(info, n, lam)
Cor.to_csv("myoutput2.2.csv", index = False)
