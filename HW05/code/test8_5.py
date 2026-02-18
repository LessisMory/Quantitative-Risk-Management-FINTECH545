import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import minimize
from scipy.stats import kurtosis


def t_es(data, alpha = 0.05):
    X = data.copy()
    nu, mu, scale = t.fit(X)
    t_alpha = t.ppf(alpha, df=nu)
    es = -mu + scale * (nu + t_alpha**2) / (nu - 1) * t.pdf(t_alpha, df=nu) / alpha
    return es


def t_es_from_mean(data, alpha = 0.05):
    X = data.copy()
    X = X - X.mean()
    nu, mu, scale = t.fit(X)
    t_alpha = t.ppf(alpha, df=nu)
    es = -mu + scale * (nu + t_alpha**2) / (nu - 1) * t.pdf(t_alpha, df=nu) / alpha
    return es


data = pd.read_csv("test7_2.csv")
alpha = 0.05
abs_es = t_es(data, alpha)
es_from_mean = t_es_from_mean(data, alpha)
sheet = pd.DataFrame(
    [{
        "ES Absolute" : abs_es,
        "ES Diff from Mean" : es_from_mean,
    }]
)
sheet.to_csv("myoutput8_5.csv", index=False)
