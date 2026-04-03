import pandas as pd
import numpy as np
from scipy.optimize import minimize

# risk parity with custom weight
def risk_parity_cw(data: pd.DataFrame):
    data = data.copy()
    covar = data.values

    risk_shares = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
    target_budgets = risk_shares / np.sum(risk_shares)

    # Function for Portfolio Volatility
    def pvol(w):
        return np.sqrt(w.T @ covar @ w)

    # Function for Component Standard Deviation
    def pCSD(w):
        pVol = pvol(w)
        # w * (covar @ w) does element-wise multiplication naturally in Python
        csd = w * (covar @ w) / pVol
        return csd

    # Sum Square Error of cSD
    def custom_sseCSD(w):
        pVol = pvol(w)
        csd = pCSD(w)
        target_csd = pVol * target_budgets # <--- THE MAGIC HAPPENS HERE
        return 1.0e5 * np.sum((csd - target_csd)**2)

    # EXECUTE OPTIMIZATION
    n = len(covar)
    bounds = [(0, None) for _ in range(n)] # Weights >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0} # Sum of weights = 1
    initial_w = np.ones(n) / n # Start with 1/n equal weights

    # SLSQP is Python's standard constrained non-linear solver
    res = minimize(custom_sseCSD, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)

    w_rp = res.x / np.sum(res.x) # Normalize just in case

    RPWeights = pd.DataFrame({'W': w_rp})
    return RPWeights

data = pd.read_csv("test5_2.csv")
W = risk_parity_cw(data)
W.to_csv("myoutput10_2.csv", index=False)