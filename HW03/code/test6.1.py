import numpy as np
import pandas as pd

data = pd.read_csv("test6.csv", index_col=0)
data.isnull().sum()
arc_r = data.pct_change(axis=0).dropna()
arc_r.to_csv("myoutput6.1.csv", index=True)


