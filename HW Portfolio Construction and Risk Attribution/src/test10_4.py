import pandas as pd
import numpy as np
from scipy.optimize import minimize


def Max_SR_W(StockMeans: pd.DataFrame, Covar: pd.DataFrame, rf):
    stockMeans = StockMeans.values
    covar = Covar.values
    # Optimize for Max Sharpe Ratio
    def sr(w):
        m = (w @ stockMeans) - rf
        s = np.sqrt(w @ covar @ w)
        # scipy only minimizes, so we return the negative Sharpe
        return -(m / s)

    n = len(stockMeans)
    bounds = [(0.1, 0.5) for _ in range(n)] # 0.1 <= Weights <= 0.5 
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0} # Sum of weights = 1
    initial_w = np.ones(n) / n

    res = minimize(sr, initial_w, bounds=bounds, constraints=constraints)
    w = res.x / np.sum(res.x) # Normalize just in case
    
    SRWeights = pd.DataFrame({"W": w})
    return SRWeights

rf = 0.04
StockMeans = pd.read_csv("test10_3_means.csv")
Covar = pd.read_csv("test5_2.csv")
SRW = Max_SR_W(StockMeans, Covar, rf)
SRW.to_csv("myoutput10_4.csv", index=False)