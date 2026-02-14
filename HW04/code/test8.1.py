import numpy as np
import pandas as pd
from scipy import stats


data = pd.read_csv("test7_1.csv")
n = data.shape[0]
mu, sigma = stats.norm.fit(data)
sigma = sigma * np.sqrt(n / (n - 1))
alpha = 0.05
VaR_1 = - (stats.norm.ppf(alpha, loc=0, scale=1) * sigma + mu)
VaR_2 = - stats.norm.ppf(alpha, loc=0, scale=1) * sigma
sheet = pd.DataFrame(
    [{
        "VaR Absolute" : VaR_1,
        "VaR Diff from Mean" : VaR_2,
    }]
)
sheet.to_csv("myoutput8_1.csv",index=False)