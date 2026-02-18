import numpy as np
import pandas as pd
from scipy.stats import norm, t, spearmanr, multivariate_normal


class Asset:
    def __init__(self, name, holding, price, dist_type):
        self._name = name
        self._holding = holding
        self._price = price
        self._dist_type = dist_type


def non_p_var_es(data, alpha = 0.05):
    """
    Empirically get VaR and ES
    """
    x = data.copy()
    x = np.sort(np.asarray(x))
    N = len(x)
    alpha_idx = N * alpha
    alpha_idx_ceil = int(np.ceil(alpha_idx)) - 1
    alpha_idx_floor = int(np.floor(alpha_idx)) - 1
    
    #Protection mechanism
    alpha_idx_ceil = max(alpha_idx_ceil, 0)
    alpha_idx_floor = max(alpha_idx_floor, 0)

    VaR = (x[alpha_idx_ceil] + x[alpha_idx_floor]) / 2
    ES = np.mean(x[0 : alpha_idx_ceil + 1])
    
    return -VaR, -ES


assets = pd.read_csv("test9_1_portfolio.csv")
assets_list = []
for _, row in assets.iterrows():
    asset = Asset(row["Stock"], row["Holding"], row["Starting Price"], row["Distribution"])
    assets_list.append(asset)

r = pd.read_csv("test9_1_returns.csv")
A = r["A"].values
B = r["B"].values
mu, sigma = norm.fit(A)
nu, loc, scale = t.fit(B)
marginal_A = norm(loc=mu, scale=sigma)
marginal_B = t(df=nu, loc=loc, scale=scale)
U = np.column_stack((marginal_A.cdf(A), marginal_B.cdf(B)))
Z = norm.ppf(U)
corr = np.corrcoef(Z, rowvar=False)

nSim = 100_000
mean = np.zeros(Z.shape[1])
Gaussian_copula = multivariate_normal(mean=mean, cov=corr)
Z = Gaussian_copula.rvs(size=nSim, random_state=42)
U = norm.cdf(Z)
r_sims = np.zeros_like(U)
r_sims[:,0] = marginal_A.ppf(U[:,0])
r_sims[:,1] = marginal_B.ppf(U[:,1])
var_a_pct, es_a_pct = non_p_var_es(r_sims[:,0])
var_b_pct, es_b_pct = non_p_var_es(r_sims[:,1])

p_sims = np.zeros_like(U)
p_sims[:,0] = r_sims[:,0] * assets_list[0]._holding * assets_list[0]._price
p_sims[:,1] = r_sims[:,1] * assets_list[1]._holding * assets_list[1]._price
var_a_p, es_a_p = non_p_var_es(p_sims[:,0])
var_b_p, es_b_p = non_p_var_es(p_sims[:,1])

Portfolio = np.zeros_like(U)
Values_in_total = assets_list[0]._holding * assets_list[0]._price + assets_list[1]._holding * assets_list[1]._price
w_A = (assets_list[0]._holding * assets_list[0]._price) / Values_in_total
w_B = (assets_list[1]._holding * assets_list[1]._price) / Values_in_total
Portfolio[:,1] = r_sims[:,0] * w_A + r_sims[:,1] * w_B
Portfolio[:,0] = Portfolio[:,1] * Values_in_total
var_portfolio_p, es_portfolio_p = non_p_var_es(Portfolio[:,0])
var_portfolio_pct, es_portfolio_pct = non_p_var_es(Portfolio[:,1])

sheet = pd.DataFrame(
    [{
        "Stock": assets_list[0]._name,
        "VaR95": var_a_p,
        "ES95": es_a_p,
        "VaR95_Pct": var_a_pct,
        "ES95_Pct": es_a_pct
    }
    ,
    {
        "Stock": assets_list[1]._name,
        "VaR95": var_b_p,
        "ES95": es_b_p,
        "VaR95_Pct": var_b_pct,
        "ES95_Pct": es_b_pct
    }
    ,
    {
        "Stock": "Portfolio",
        "VaR95": var_portfolio_p,
        "ES95": es_portfolio_p,
        "VaR95_Pct": var_portfolio_pct,
        "ES95_Pct": es_portfolio_pct    
    }]
)
sheet.to_csv("myoutput9_1.csv", index=False)



