import numpy as np
import pandas as pd


def weight_norm_helper(A, W):
    """
    W : weight vector (n,)
    """
    weighted_A = np.diag(np.sqrt(W)) @ A @ np.diag(np.sqrt(W))
    return np.linalg.norm(weighted_A, ord="fro")

def ps_helper(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def pu_helper(A):
    pu_A = A.copy()
    np.fill_diagonal(pu_A, 1.0)
    return pu_A

# def fronorm_cal(A, W):
#     W = W.asarray().reshape(-1)
#     #np.sum((W[:,None] * A * W[None, :]) ** 2)
#     A_ = W @ A @ W
#     return np.linalg.norm(A_,ord="fro")

def higham_Cov(Cov, tolerance, W, Maxtimes):
    Cov_mat = Cov.copy()
    Cov_mat = (Cov_mat + Cov_mat.T) / 2
    iv_sd = np.diag(1.0 / np.sqrt(np.diag(Cov_mat)))
    pc = iv_sd @ Cov_mat @ iv_sd
    Y = pc.copy()
    delta = np.zeros_like(Y)
    diff = np.inf
    i = 1
    while i <= Maxtimes:
        R = Y - delta
        X = ps_helper(R)
        delta = X - R
        Y_new = pu_helper(X)
        diff_new = weight_norm_helper(Y_new - pc, W)
        min_eig = np.linalg.eigvals(Y_new).min()
        if np.abs(diff_new - diff) < tolerance and min_eig > -1e-9:
            break
        Y = Y_new
        diff = diff_new
        i += 1
    sd = np.diag(1.0 / np.diag(iv_sd))
    ad_Cov_mat = sd @ Y @ sd
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


def sim_higham_Cov(times, Cov_mat, mu, tolerance, W, Maxtimes):
    reng = np.random.default_rng(42)
    vars = Cov_mat.shape[0]
    ad_Cov = higham_Cov(Cov_mat, tolerance, W, Maxtimes)
    L = chol_psd(ad_Cov)
    Z = reng.normal(loc=0, scale=1, size=(vars, times))
    X = mu + L @ Z
    sim_Cov_mat = np.cov(X)
    return sim_Cov_mat


ntimes = 100_000
mu = 0
Cov_data = pd.read_csv("test5_3.csv")
Cov_mat = Cov_data.to_numpy()
tolerance = 1e-9
n = Cov_mat.shape[0]
W = np.ones(n)
Maxtimes = 100
sim_Cov_mat = sim_higham_Cov(ntimes, Cov_mat, mu, tolerance, W, Maxtimes)
output = pd.DataFrame(sim_Cov_mat, columns=Cov_data.columns)
output.to_csv("myoutput5_4.csv", index=False)