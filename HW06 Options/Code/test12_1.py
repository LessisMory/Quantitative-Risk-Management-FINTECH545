import pandas as pd
import numpy as np
from scipy.stats import norm


def calculate_greeks_vectorized(df):
    """
    Calculates Price and Greeks for a complete Pandas DataFrame instantly.
    Expects columns matching the user's dataset.
    """

    df = df.copy()
    df = df.dropna()
    df['ID'] = df['ID'].astype(int)

    # 1. Extract columns as fast numpy arrays
    S = df['Underlying'].values
    X = df['Strike'].values
    T = (df['DaysToMaturity'] / df['DayPerYear']).values 
    r = df['RiskFreeRate'].values
    q = df['DividendRate'].values
    sigma = df['ImpliedVol'].values
    is_call = (df['Option Type'] == 'Call').values
    b = r - q

    d1 = (np.log(S / X) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d1 = norm.cdf(-d1)
    cdf_neg_d2 = norm.cdf(-d2)

    exp_carry = np.exp((b - r) * T)
    exp_rate = np.exp(-r * T)

    # PRICE
    call_price = S * exp_carry * cdf_d1 - X * exp_rate * cdf_d2
    put_price = X * exp_rate * cdf_neg_d2 - S * exp_carry * cdf_neg_d1
    price = np.where(is_call, call_price, put_price)

    # DELTA
    call_delta = exp_carry * cdf_d1
    put_delta = exp_carry * (cdf_d1 - 1.0)
    delta = np.where(is_call, call_delta, put_delta)

    # THETA
    term1 = -(S * sigma * exp_carry * pdf_d1) / (2 * np.sqrt(T))
    call_theta = term1 - (b - r) * S * exp_carry * cdf_d1 - r * X * exp_rate * cdf_d2
    put_theta = term1 + (b - r) * S * exp_carry * cdf_neg_d1 + r * X * exp_rate * cdf_neg_d2
    theta_annual = np.where(is_call, call_theta, put_theta)
    theta = theta_annual

    # RHO
    call_rho = T * X * exp_rate * cdf_d2
    put_rho = -T * X * exp_rate * cdf_neg_d2
    rho = np.where(is_call, call_rho, put_rho)

    # GAMMA & VEGA
    gamma = (pdf_d1 * exp_carry) / (S * sigma * np.sqrt(T))
    vega = (S * exp_carry * pdf_d1 * np.sqrt(T))

    results_df = pd.DataFrame({
        'Value': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }, index=df.ID)
    
    return results_df

data = pd.read_csv("test12_1.csv")
db = calculate_greeks_vectorized(data)
db.to_csv("myoutput12_1.csv", index=True)


