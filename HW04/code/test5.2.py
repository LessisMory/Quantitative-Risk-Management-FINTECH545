import numpy as np
import pandas as pd


def chol_psd(input):
    n = input.shape[0]
    out_mat = np.zeros((n, n))
    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(out_mat[j, :j], out_mat[j, :j])
        temp = input[j, j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        out_mat[j, j] = np.sqrt(temp) 
        if out_mat[j, j] != 0.0:
            ivdiag = 1.0 / out_mat[j, j]
            for i in range(j+1, n):
                s = np.dot(out_mat[i, :j], out_mat[j, :j])
                out_mat[i, j] = (input[i, j] - s) * ivdiag   
    return out_mat


def simulate_PSD_Cov(times, Cov_mat, mu):
    vars = Cov_mat.shape[0]
    reng = np.random.default_rng(42)
    L = chol_psd(Cov_mat)
    Z = reng.normal(loc = 0, scale = 1, size = (vars, times))
    X = mu + L @ Z
    si_Cov_mat = np.cov(X)
    return si_Cov_mat


ntimes = 100_000
mu = 0
Cov_data = pd.read_csv("test5_2.csv")
Cov_mat = Cov_data.to_numpy()
si_Cov_mat = simulate_PSD_Cov(ntimes, Cov_mat, mu)
output = pd.DataFrame(si_Cov_mat, columns=Cov_data.columns)
output.to_csv("myoutput5_2.csv", index=False)
