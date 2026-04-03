import pandas as pd
import numpy as np
from scipy.optimize import minimize


def Expost_Attribution(StockReturn, StockWeights):
    # Calculate daily returns from prices (Equivalent to return_calculate in Julia)
    stocks = [ c for c in stocks_ret.columns]
    n_days = ret.shape[0]
    num_stocks = w.shape[0]
    pReturn = np.zeros(n_days)
    weights = np.zeros((n_days, num_stocks))
    lastW = w.squeeze()
    matReturns = ret

    # Simulating the drift
    for i in range(n_days):
        weights[i, :] = lastW
        lastW = lastW * (1.0 + matReturns[i, :])
        pR = np.sum(lastW)
        lastW = lastW / pR
        pReturn[i] = pR - 1

    stocks_ret['Portfolio'] = pReturn

    # 5. Cariño's K & Ex-Post Return Attribution
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    k = np.log(totalRet + 1) / totalRet
    carinoK = np.log(1.0 + pReturn) / pReturn / k

    # Calculate scaled attribution per day
    attrib = pd.DataFrame(matReturns * weights * carinoK[:, None], columns=stocks)

    # Set up the Attribution output dataframe
    tr_row = []
    attr_row = []
    cols = stocks + ['Portfolio']

    for s in cols:
        # Total geometric return over the period
        tr = np.exp(np.sum(np.log(stocks_ret[s] + 1))) - 1
        # Attribution sum
        atr = attrib[s].sum() if s != 'Portfolio' else tr
        
        tr_row.append(tr)
        attr_row.append(atr)

    attribution = pd.DataFrame([tr_row, attr_row], columns=cols)
    attribution.insert(0, 'Value', ['Total Return', 'Return Attribution'])

    # 6. Realized Volatility Attribution
    Y_vol = matReturns * weights
    X_vol = np.column_stack((np.ones(n_days), pReturn))

    # OLS Regression for Risk Beta
    B_vol = (np.linalg.inv(X_vol.T @ X_vol) @ X_vol.T @ Y_vol)[1, :] 

    # Component SD: Beta * std(Portfolio)
    # Note: Julia std uses ddof=1 by default, so we specify it in numpy
    pStd = np.std(pReturn, ddof=1)
    cSD = B_vol * pStd

    # Add the Vol attribution to the output
    vol_row = ['Vol Attribution'] + list(cSD) + [pStd]
    attribution.loc[2] = vol_row
    return attribution

stocks_ret = pd.read_csv("test11_1_returns.csv")
ret = stocks_ret.values
stocks_w = pd.read_csv("test11_1_weights.csv")
w = stocks_w.values
Attribution = Expost_Attribution(ret, w)
Attribution.to_csv("myoutput11_1.csv", index=False)