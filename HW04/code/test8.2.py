import numpy as np
import pandas as pd
from scipy import stats


data = pd.read_csv("test7_2.csv")
n = data.shape[0]
df, loc, scale = stats.t.fit(data)
alpha = 0.05
VaR_1 = - (stats.t.ppf(alpha,df,loc=0,scale=1) * scale + loc)
VaR_2 = - stats.t.ppf(alpha,df,loc=0,scale=1) * scale
sheet = pd.DataFrame(
    [{
        "VaR Absolute" : VaR_1,
        "VaR Diff from Mean" : VaR_2,
    }]
)
sheet.to_csv("myoutput8_2.csv",index=False)