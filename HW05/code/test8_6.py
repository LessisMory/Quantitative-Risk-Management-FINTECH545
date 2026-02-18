import numpy as np
import pandas as pd
from scipy.stats import t
from test8_5 import t_es, t_es_from_mean

def monte_carlo_es(data, alpha=0.05, nSim = 100_000):
    X = data.copy()
    nu, mu, scale = t.fit(X)
    t_dist = t(df=nu, loc=mu, scale=scale)
    t_sims = t_dist.rvs(size=nSim, random_state=42)
    t_sims.sort()
    index_ceil = int(np.ceil(nSim * alpha)) - 1
    es = np.mean(t_sims[:index_ceil+1])
    # es = t_es(t_sims)
    # return es
    return -es


def monte_carlo_es_from_mean(data, alpha=0.05, nSim = 100_000):
    # X = data.copy()
    # nu, mu, scale = t.fit(X)
    # t_dist = t(df=nu, loc=mu, scale=scale)
    # t_sims = t_dist.rvs(size=nSim, random_state=42)
    # es = t_es_from_mean(t_sims)
    # return es 
    X = data.copy()
    X = X - X.mean()
    nu, mu, scale = t.fit(X)
    t_dist = t(df=nu, loc=mu, scale=scale)
    t_sims = t_dist.rvs(size=nSim, random_state=42)
    t_sims.sort()
    index_ceil = int(np.ceil(nSim * alpha)) - 1
    es = np.mean(t_sims[:index_ceil+1])
    return -es


data = pd.read_csv("test7_2.csv")
abs_es = monte_carlo_es(data)
es_from_mean = monte_carlo_es_from_mean(data)
sheet = pd.DataFrame(
    [{
        "ES Absolute" : abs_es,
        "ES Diff from Mean" : es_from_mean,
    }]
)
sheet.to_csv("myoutput8_6.csv", index=False)
