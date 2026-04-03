import pandas as pd
import numpy as np
from scipy.optimize import minimize


def ex_post_fators(stocks_ret, stocks_w, factors_ret, factor_beta):
    s_ret = stocks_ret.values
    w = stocks_w.squeeze()
    f_ret = factors_ret.values
    beta = factor_beta.iloc[:,1:].values

    factors = [ f for f in factors_ret]
    n_days = s_ret.shape[0]
    pReturn = np.zeros(n_days)
    residReturn = np.zeros(n_days)
    factorWeights = np.zeros((n_days, len(factors)))
    lastW = w.copy()

    matReturns = s_ret
    ffReturns = f_ret

    for i in range(n_days):
        # Calculate Factor Weights (w @ Betas)
        factorWeights[i, :] = lastW @ beta
        
        # Update Stock Weights
        lastW = lastW * (1.0 + matReturns[i, :])
        pR = np.sum(lastW)
        lastW = lastW / pR
        pReturn[i] = pR - 1
        
        # Calculate Residual (Alpha) = Total Return - Factor Return
        residReturn[i] = pReturn[i] - (factorWeights[i, :] @ ffReturns[i, :])

    # --- 4. RETURN & RISK ATTRIBUTION ---
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    k = np.log(totalRet + 1) / totalRet
    carinoK = np.log(1.0 + pReturn) / pReturn / k

    # Scale Factor Returns & Alpha by Cariño's K
    attrib = pd.DataFrame(ffReturns * factorWeights * carinoK[:, None], columns=factors)
    attrib['Alpha'] = residReturn * carinoK

    newFactors = factors + ['Alpha']
    factors_ret['Alpha'] = residReturn
    factors_ret['Portfolio'] = pReturn

    tr_row = []
    attr_row = []
    cols = newFactors + ['Portfolio']

    for s in cols:
        tr = np.exp(np.sum(np.log(factors_ret[s] + 1))) - 1 if s in factors_ret else np.nan
        atr = attrib[s].sum() if s != 'Portfolio' else tr
        tr_row.append(tr)
        attr_row.append(atr)

    attribution = pd.DataFrame([tr_row, attr_row], columns=cols)
    attribution.insert(0, 'Value', ['Total Return', 'Return Attribution'])

    # Risk Attribution via Regression
    Y_vol = np.column_stack((ffReturns * factorWeights, residReturn))
    X_vol = np.column_stack((np.ones(n_days), pReturn))

    B_vol = (np.linalg.inv(X_vol.T @ X_vol) @ X_vol.T @ Y_vol)[1, :]
    pStd = np.std(pReturn, ddof=1)
    cSD = B_vol * pStd

    vol_row = ['Vol Attribution'] + list(cSD) + [pStd]
    attribution.loc[2] = vol_row
    return attribution


stocks_ret = pd.read_csv("test11_2_stock_returns.csv")
stocks_w = pd.read_csv("test11_2_weights.csv")
factors_ret = pd.read_csv("test11_2_factor_returns.csv")
factor_beta = pd.read_csv("test11_2_beta.csv")
outputForm = ex_post_fators(stocks_ret, stocks_w, factors_ret, factor_beta)
outputForm.to_csv("myoutput11_2.csv",index=False)