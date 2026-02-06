import pandas as pd
data = pd.read_csv("test1.csv")
data = data.dropna()
CovMatrix = data.cov()
CovMatrix.to_csv("myoutput1.1.csv", index=False)