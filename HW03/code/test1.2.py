import pandas as pd
data = pd.read_csv("test1.csv")
data = data.dropna()
CorMatrix = data.corr()
CorMatrix.to_csv("myoutput1.2.csv", index=False)