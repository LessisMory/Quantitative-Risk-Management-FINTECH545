import pandas as pd
import numpy as np

def bt_american(is_call, S, K, T, r, b, sigma, N):
    """
    Prices an American Option using a Recombining Binomial Tree.
    """
    # 1. Setup the CRR lattice parameters
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    z = 1 if is_call else -1

    # 2. Setup the expiration nodes (j = N)
    # np.arange(N + 1) creates an array of up-moves: [0, 1, 2, ..., N]
    i_moves = np.arange(N + 1)
    
    # Vectorized calculation of all terminal stock prices
    prices = S * (u ** i_moves) * (d ** (N - i_moves))
    
    # Terminal option values (Intrinsic payoff)
    option_values = np.maximum(0, z * (prices - K))

    # 3. Backward Induction Loop
    for j in range(N - 1, -1, -1):
        # Calculate the stock prices at this earlier time step 'j'
        i_moves = np.arange(j + 1)
        prices = S * (u ** i_moves) * (d ** (j - i_moves))
        
        # Calculate the Intrinsic Payoff if exercised right now
        intrinsic = np.maximum(0, z * (prices - K))
        
        # Calculate the Continuation Value
        # option_values[1:j+2] are the "Up" nodes from the future step
        # option_values[0:j+1] are the "Down" nodes from the future step
        continuation = df * (pu * option_values[1:j+2] + pd * option_values[0:j+1])
        
        # The American Decision: Max of Dead or Alive
        # This overwrites the array in-place, slicing off the top node each step!
        option_values = np.maximum(intrinsic, continuation)

    # When the loop finishes at j=0, only the present value remains
    return option_values[0]

def calculate_row_greeks(row):
    ID = int(row['ID'])
    is_call = True if str(row['Option Type']).strip().lower() == 'call' else False
    S = float(row['Underlying'])
    K = float(row['Strike'])
    T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    r = float(row['RiskFreeRate'])
    q = float(row['DividendRate'])
    sigma = float(row['ImpliedVol'])
    N = 500
    #N = int(row['DaysToMaturity'])
    
    b = r - q 
    price = bt_american(is_call, S, K, T, r, b, sigma, N)
    
    dS = S * 0.01  
    dSig = 0.01  
    dR = 0.01  
    dT = 1 / 365.0

    # Delta & Gamma
    p_up_s = bt_american(is_call, S + dS, K, T, r, b, sigma, N)
    p_dn_s = bt_american(is_call, S - dS, K, T, r, b, sigma, N)
    
    fd_delta = (p_up_s - p_dn_s) / (2 * dS)
    fd_gamma = (p_up_s - 2 * price + p_dn_s) / (dS ** 2)

    # Vega
    p_up_sig = bt_american(is_call, S, K, T, r, b, sigma + dSig, N)
    p_dn_sig = bt_american(is_call, S, K, T, r, b, sigma - dSig, N)
    vega = (p_up_sig - p_dn_sig) / (2 * dSig)

    # Rho    
    p_up_r = bt_american(is_call, S, K, T, r + dR, b, sigma, N)
    p_dn_r = bt_american(is_call, S, K, T, r - dR, b, sigma, N)
    
    rho = (p_up_r - p_dn_r) / (2 * dR)
    
    # Theta
    if T - dT > 0:
        p_shorter_t = bt_american(is_call, S, K, T - dT, r, b, sigma, N)
        theta = (price - p_shorter_t) / dT 
    else:
        theta = 0.0

    return pd.Series({
        'ID': ID,
        'Value': price,
        'Delta': fd_delta,
        'Gamma': fd_gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    })

data = pd.read_csv("test12_1.csv").dropna()
results_df = data.apply(lambda row: calculate_row_greeks(row), axis=1)
results_df['ID'] = results_df['ID'].astype(int)
results_df.to_csv("myoutput12_2.csv", index=False)