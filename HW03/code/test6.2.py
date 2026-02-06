import numpy as np
import pandas as pd

data = pd.read_csv("test6.csv", index_col=0)
data.isnull().sum()
log_r = np.log(data).diff().dropna()
log_r.to_csv("myoutput6.2.csv", index=True)