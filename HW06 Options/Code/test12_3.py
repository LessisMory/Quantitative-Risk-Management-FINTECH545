import pandas as pd
import numpy as np

data = pd.read_csv("test12_3.csv")

def bt_american_discrete_div(is_call, S, K, T, r, div_amts, div_times, sigma, N):
    """
    Prices an American Option with Discrete Dividends using a Recursive Binomial Tree.
    
    Parameters:
    div_amts (list/array): The cash amounts of the dividends.
    div_times (list/array): The time steps (j) when the dividends are paid.
    """
    # Ensure inputs are numpy arrays for easy mathematical slicing
    div_amts = np.array(div_amts)
    div_times = np.array(div_times)
    
    # --- PHASE 1: The Base Case (The Escape Hatch) ---
    # If there are no dividends left, or the next dividend happens AFTER expiration,
    # we just run the standard, highly-optimized recombining tree.
    if len(div_amts) == 0 or len(div_times) == 0 or div_times[0] > N:
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        pu = (np.exp(r * dt) - d) / (u - d) # Cost of carry b = r for discrete divs
        pd = 1.0 - pu
        df = np.exp(-r * dt)
        z = 1 if is_call else -1

        i_moves = np.arange(N + 1)
        prices = S * (u ** i_moves) * (d ** (N - i_moves))
        option_values = np.maximum(0, z * (prices - K))

        for j in range(N - 1, -1, -1):
            i_moves = np.arange(j + 1)
            prices = S * (u ** i_moves) * (d ** (j - i_moves))
            intrinsic = np.maximum(0, z * (prices - K))
            continuation = df * (pu * option_values[1:j+2] + pd * option_values[0:j+1])
            option_values = np.maximum(intrinsic, continuation)

        return option_values[0]

    # --- PHASE 2: The Inception Point (Handling the Discontinuity) ---
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(r * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    z = 1 if is_call else -1

    n_div = div_times[0] # The exact step 'j' where the first dividend hits
    
    # Allocate a 1D array strictly for the size of the tree at the dividend date
    option_values = np.zeros(n_div + 1)
    
    # Calculate all possible stock prices right BEFORE the dividend is paid
    i_moves = np.arange(n_div + 1)
    prices_pre_div = S * (u ** i_moves) * (d ** (n_div - i_moves))
    
    # We must loop through these specific nodes because each one spawns its own tree
    for i in range(n_div + 1):
        price = prices_pre_div[i]
        
        # Dead Value: Exercise the option right before the stock drops
        val_exercise = max(0, z * (price - K))
        
        # Alive Value: Hold the option. 
        # RECURSION: We spawn a brand new tree starting from the post-drop price!
        val_hold = bt_american_discrete_div(
            is_call, 
            S=price - div_amts[0],             # Stock price takes the dividend hit
            K=K, 
            T=T - (n_div * dt),                # Time remaining shrinks
            r=r, 
            div_amts=div_amts[1:],             # Remove the dividend we just paid
            div_times=div_times[1:] - n_div,   # Shift remaining dividend times forward
            sigma=sigma, 
            N=N - n_div                        # Steps remaining shrinks
        )
        
        # The American Decision at the dividend cliff
        option_values[i] = max(val_exercise, val_hold)

    # --- PHASE 3: Backward Induction ---
    # Now that we have the resolved values at the dividend cliff, we step 
    # backward normally to today (j = 0) using our fast vectorization.
    for j in range(n_div - 1, -1, -1):
        i_moves = np.arange(j + 1)
        prices = S * (u ** i_moves) * (d ** (j - i_moves))
        
        intrinsic = np.maximum(0, z * (prices - K))
        continuation = df * (pu * option_values[1:j+2] + pd * option_values[0:j+1])
        option_values = np.maximum(intrinsic, continuation)

    return option_values[0]

def calculate_discrete_div_price(row):
    is_call = True if str(row['Option Type']).strip().lower() == 'call' else False
    S = float(row['Underlying'])
    K = float(row['Strike'])
    T = float(row['DaysToMaturity']) / float(row['DayPerYear'])
    r = float(row['RiskFreeRate'])
    sigma = float(row['ImpliedVol'])
    
    N = int(row['DaysToMaturity'])
    
    if pd.isna(row['DividendDates']) or str(row['DividendDates']).strip() == "":
        div_times = []
    else:
        div_times = [int(float(x)) for x in str(row['DividendDates']).split(',')]
        
    if pd.isna(row['DividendAmts']) or str(row['DividendAmts']).strip() == "":
        div_amts = []
    else:
        div_amts = [float(x) for x in str(row['DividendAmts']).split(',')]
        
    price = bt_american_discrete_div(
        is_call=is_call, 
        S=S, 
        K=K, 
        T=T, 
        r=r, 
        div_amts=div_amts, 
        div_times=div_times, 
        sigma=sigma, 
        N=N
    )
    
    return price

data['Value'] = data.apply(calculate_discrete_div_price, axis=1)
export_df = data[["ID","Price"]]
export_df.to_csv("myoutput12_3.csv", index=False)