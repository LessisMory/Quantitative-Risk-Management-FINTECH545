import pandas as pd
import numpy as np
data = pd.read_csv("test1.csv")
n = data.shape[1]
Cor = np.zeros((n, n))
for i in range(n):
    for j in range(i+1):
        col_i = data.columns[i]
        col_j = data.columns[j]
        # std_i = data[col_i].dropna().std(ddof=1)
        # std_j = data[col_j].dropna().std(ddof=1)
        if i == j:
            Cor[i][j] = 1
        else:
            valid_row = data[col_i].notna() & data[col_j].notna()
            std_i = data.loc[valid_row, col_i].std(ddof=1)
            std_j = data.loc[valid_row, col_j].std(ddof=1)
            Cor[i][j] = np.cov(data.loc[valid_row, col_i], data.loc[valid_row, col_j], ddof=1)[0,1] / (std_i * std_j)
            Cor[j][i] = Cor[i][j]
Cor = pd.DataFrame(Cor, columns=data.columns)
Cor.to_csv("myoutput1.4.csv", index=False)

        