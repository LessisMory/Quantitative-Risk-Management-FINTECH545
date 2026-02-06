import pandas as pd
import numpy as np

data = pd.read_csv("testout_1.3.csv")
Cov_mat = data.to_numpy(dtype = float)

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

Cov_mat_ad = near_psd_cov(Cov_mat)
Cov_mat_ad = pd.DataFrame(Cov_mat_ad, columns=data.columns)
Cov_mat_ad.to_csv("myoutput3.1.csv", index=False)
