import pandas as pd
import numpy as np
from scipy import stats

def simulate_normal(input_csv: str, output_csv: str):
    Data = pd.read_csv(input_csv)
    data = Data.iloc[:,0].to_numpy()
    mu, sigma = stats.norm.fit(data)
    sigma_adj = sigma * np.sqrt(data.size / (data.size - 1)) #unbiased but not MLE anymore
    sheet = pd.DataFrame(
        [{
            "mu" : mu,
            "sigma" : sigma,
            "unbiased sigma" : sigma_adj
        }]
    )
    sheet.to_csv(output_csv, index = False)

def main():
    simulate_normal("HW01/test7_1.csv","HW01/output7_1.csv")

if __name__ == "__main__":
    main()