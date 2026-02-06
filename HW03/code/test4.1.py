import pandas as pd
import numpy as np

data = pd.read_csv("testout_3.1.csv")
Cov_mat = data.to_numpy(dtype = float)

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

sm_L = chol_psd(Cov_mat)
sm_L = pd.DataFrame(sm_L, columns = data.columns)
sm_L.to_csv("myoutput4.1.csv",index=False)
