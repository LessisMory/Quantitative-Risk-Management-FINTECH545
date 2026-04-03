import pandas as pd
import numpy as np
from scipy.optimize import minimize

def risk_parity(data: pd.DataFrame):
    data = data.copy()
    covar = data.values

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
    def sseCSD(w):
        csd = pCSD(w)
        mCSD = np.mean(csd)
        dCsd = csd - mCSD
        se = dCsd ** 2
        # Add a large multiplier (1e5) for better solver convergence
        return 1.0e5 * np.sum(se)

    # EXECUTE OPTIMIZATION
    n = len(covar)
    bounds = [(0, None) for _ in range(n)] # Weights >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0} # Sum of weights = 1
    initial_w = np.ones(n) / n # Start with 1/n equal weights

    # SLSQP is Python's standard constrained non-linear solver
    res = minimize(sseCSD, initial_w, method='SLSQP', bounds=bounds, constraints=constraints)

    w_rp = res.x / np.sum(res.x) # Normalize just in case

    RPWeights = pd.DataFrame({'W': w_rp})
    return RPWeights

data = pd.read_csv("test5_2.csv")
W = risk_parity(data)
W.to_csv("myoutput10_1.csv", index=False)