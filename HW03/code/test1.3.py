import pandas as pd
import numpy as np
data = pd.read_csv("test1.csv")
n = data.shape[1]
Cov = np.zeros((n, n))
for i in range(n):
    for j in range(i+1):
        col_i = data.columns[i]
        col_j = data.columns[j]
        if i == j:
            Cov[i][j] = data[col_i].dropna().var(ddof = 1)
        else:
            valid_row = data[col_i].notna() & data[col_j].notna()
            Cov[i][j] = np.cov(data.loc[valid_row, col_i], data.loc[valid_row, col_j], ddof = 1)[0,1]
            Cov[j][i] = Cov[i][j]
Cov = pd.DataFrame(Cov, columns = data.columns)
Cov.to_csv("myoutput1.3.csv", index=False)