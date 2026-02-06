import pandas as pd
import numpy as np

data = pd.read_csv("testout_1.4.csv")
Cor_mat = data.to_numpy(dtype = float)

def weight_norm_helper(A, W):
    """
    W : weight vector (n,)
    """
    weight_A = np.diag(np.sqrt(W)) @ A @ np.diag(np.sqrt(W))
    return np.linalg.norm(weight_A, ord="fro")

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

def higham_Cor(Cor, tolerance, W, Maxtimes):
    pc = Cor.copy()
    Y = pc
    delta = np.zeros_like(Y)
    diff = np.inf
    for _ in range(Maxtimes):
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
    return Y

n = Cor_mat.shape[0]
W = np.ones(n)
ad_Cor_mat = higham_Cor(Cor_mat, 1e-9, W, 100)
ad_Cor_mat = pd.DataFrame(ad_Cor_mat, columns=data.columns)
ad_Cor_mat.to_csv("myoutput3.4.csv", index=False)