import numpy as np
import pandas as pd


def pca_simulate(Cov_mat, times, ex_level):
    #iif Cov_mat are at least semi-postive definite
    reng = np.random.default_rng(42)
    eig_vals, eig_vecs = np.linalg.eigh(Cov_mat)
    eig_vals = eig_vals[::-1]
    eig_vecs = eig_vecs[:,::-1]
    mask = eig_vals > 0
    pos_eigvals = eig_vals[mask]
    pos_eigvecs = eig_vecs[:,mask]
    total_var = pos_eigvals.sum()
    # if nval != 0:
    #     if nval < pos_eigvals.length():
    #         pos_eigvals = pos_eigvals[:nval-1]
    cum_explained = np.cumsum(pos_eigvals) / total_var
    k = np.searchsorted(cum_explained, ex_level)
    c_eig_vals = pos_eigvals[:k+1]
    c_eig_vecs = pos_eigvecs[:,:k+1]
    B = c_eig_vecs @ np.diag(np.sqrt(c_eig_vals))
    m = c_eig_vals.shape[0]
    r = reng.normal(loc = 0, scale = 1, size=(m,times))
    X = (B @ r).T
    return X


ntimes = 100_000
ex_level = 0.99
Cov_data = pd.read_csv("test5_2.csv")
Cov_mat = Cov_data.to_numpy()
sim_data = pca_simulate(Cov_mat, ntimes, ex_level)
sim_Cov_mat = np.cov(sim_data.T)
output = pd.DataFrame(sim_Cov_mat, columns=Cov_data.columns)
output.to_csv("myoutput5_5.csv", index=False)
