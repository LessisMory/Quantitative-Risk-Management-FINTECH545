import numpy as np
import pandas as pd
import math
from scipy import stats


data = pd.read_csv("test7_2.csv")
df, loc, scale = stats.t.fit(data)
ntimes = 100_000
reng = np.random.default_rng(42)
sim_returns = reng.standard_t(df, size=ntimes) * scale + loc
sim_returns.sort()
alpha_index = int(math.floor(0.05 * ntimes))
q = sim_returns[alpha_index]
mu = np.mean(data)
sheet = pd.DataFrame(
    [{
        "VaR Absolute" : -q,
        "VaR Diff from Mean" : mu - q,
    }]
)
sheet.to_csv("myoutput8_3.csv", index=False)