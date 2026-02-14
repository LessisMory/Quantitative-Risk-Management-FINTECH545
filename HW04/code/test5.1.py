import numpy as np
import pandas as pd


def simulation_PD_Cov(times, Cov_mat, mu):
    reng = np.random.default_rng(42)
    vars = Cov_mat.shape[0]
    L = np.linalg.cholesky(Cov_mat)
    Z = reng.normal(loc = 0, scale = 1, size = (vars, times))
    X = mu + L @ Z
    si_Cov_mat = np.cov(X)
    return si_Cov_mat


ntimes = 100_000
Cov_data = pd.read_csv("test5_1.csv")
Cov_mat = Cov_data.to_numpy()
si_Cov_mat = simulation_PD_Cov(ntimes, Cov_mat, 0)
output = pd.DataFrame(si_Cov_mat, columns=Cov_data.columns)
output.to_csv("myoutput5_1.csv", index=False)



