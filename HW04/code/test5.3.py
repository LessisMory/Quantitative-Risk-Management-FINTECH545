import numpy as np
import pandas as pd


def near_psd_cov(cov, epsilon=1e-8):
    init = cov.copy()
    invSD = np.diag(1.0 / np.sqrt(np.diag(init)))
    Cor_mat = invSD @ init @ invSD
    eigvals, eigvec = np.linalg.eigh(Cor_mat)
    eigvals = np.maximum(eigvals, epsilon)
    # ad_Cor_mat = eigvec @ np.diag(eigvals) @ eigvec.T
    # scale_mat = np.diag(1.0 / np.sqrt(np.diag(ad_Cor_mat)))
    # ad_Cor_mat = scale_mat @ ad_Cor_mat @ scale_mat
    scale_vec_left = 1.0 / ((eigvec * eigvec) @ eigvals)
    scale_vec_left = np.diag(np.sqrt(scale_vec_left))
    scale_vec_right = np.diag(np.sqrt(eigvals))
    B = scale_vec_left @ eigvec @ scale_vec_right
    ad_Cor_mat = B @ B.T
    SD = np.diag(1.0 / np.diag(invSD))
    ad_Cov_mat = SD @ ad_Cor_mat @ SD
    return ad_Cov_mat


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


def sim_npd_Cov(times, Cov_mat, mu):
    reng = np.random.default_rng(42)
    vars = Cov_mat.shape[0]
    ad_Cov = near_psd_cov(Cov_mat)
    L = chol_psd(ad_Cov)
    Z = reng.normal(loc=0, scale=1, size=(vars, times))
    X = mu + L @ Z
    sim_Cov_mat = np.cov(X)
    return sim_Cov_mat


ntimes = 100_000
mu = 0
Cov_data = pd.read_csv("test5_3.csv")
Cov_mat = Cov_data.to_numpy()
sim_Cov_mat = sim_npd_Cov(ntimes, Cov_mat, mu)
output = pd.DataFrame(sim_Cov_mat, columns=Cov_data.columns)
output.to_csv("myoutput5_3.csv", index=False)
