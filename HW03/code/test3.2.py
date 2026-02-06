import pandas as pd
import numpy as np

data = pd.read_csv("testout_1.4.csv")
Cor_mat = data.to_numpy(dtype = float)

def near_psd_cor(cor, epsilon=1e-8):
    init = cor.copy()
    eigvals, eigvec = np.linalg.eigh(Cor_mat)
    eigvals = np.maximum(eigvals, epsilon)
    scale_vec_left = 1.0 / ((eigvec * eigvec) @ eigvals)
    scale_vec_left = np.diag(np.sqrt(scale_vec_left))
    scale_vec_right = np.diag(np.sqrt(eigvals))
    B = scale_vec_left @ eigvec @ scale_vec_right
    ad_Cor_mat = B @ B.T
    return ad_Cor_mat

Cor = near_psd_cor(Cor_mat)
Cor = pd.DataFrame(Cor, columns = data.columns)
Cor.to_csv("myoutput3.2.csv", index=False)