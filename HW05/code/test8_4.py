import pandas as pd
from scipy.stats import norm
import numpy as np

data = pd.read_csv("test7_1.csv")
alpha = 0.05

def Delta_Norm_ES(data, alpha):
    X = data.copy()
    mu, sigma = norm.fit(X, ddof=1)
    z = norm.ppf(alpha)
    ES = -mu + sigma * norm.pdf(z) / alpha
    return ES

def Delta_Norm_ES_from_Mean(data, alpha):
    X = data.copy()
    X = X - np.mean(X)
    mu, sigma = norm.fit(X, ddof=1)
    z = norm.ppf(alpha)
    ES = -mu + sigma * norm.pdf(z) / alpha
    return ES

abs_es = Delta_Norm_ES(data, alpha)
es_from_mean = Delta_Norm_ES_from_Mean(data, alpha)
sheet = pd.DataFrame(
    [{
        "ES Absolute" : abs_es,
        "ES Diff from Mean" : es_from_mean,
    }]
)
sheet.to_csv("myoutput8_4.csv", index=False)


                         
