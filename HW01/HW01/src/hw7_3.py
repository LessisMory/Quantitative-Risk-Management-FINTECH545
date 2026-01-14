import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.tmodel import TLinearModel

def t_regression(filename: str):
    Data = pd.read_csv(filename)
    n = Data.shape[1]
    Y = Data.iloc[:,-1].to_numpy()
    X = Data.iloc[:,:n-1].to_numpy()
    t_results = t_regression_fit(Y, X)
    t_betahat = t_results.params
    sheet = pd.DataFrame(
        [{
            "mu" : 0.0, #its a model assumption without calculation
            "sigma" : t_betahat[5],
            "degree of freedom" : t_betahat[4],
            "intercept" : t_betahat[0],
            "Beta1" : t_betahat[1],
            "Beta2" : t_betahat[2],
            "Beta3" : t_betahat[3]
        }
        ])
    sheet.to_csv("HW01/output7_3.csv", index = False)

def t_regression_fit(Y, X):
    X = sm.add_constant(X)
    t_regression = TLinearModel(Y,X)
    t_results = t_regression.fit()
    return t_results

def main():
    t_regression("HW01/test7_3.csv")

if __name__ == "__main__":
    main()