import pandas as pd
import numpy as np
from scipy import stats

def simulate_t(input_csv: str, output_csv: str):
    Data = pd.read_csv(input_csv)
    data = Data.iloc[:,0].to_numpy()
    df, loc, scale = stats.t.fit(data)
    sheet = pd.DataFrame(
        [{
            "mu" : loc,
            "sigma" : scale,
            "degree of freedom" : df
        }]
    )
    sheet.to_csv(output_csv, index = False)

def main():
    simulate_t("HW01/test7_2.csv", "HW01/output7_2.csv")

if __name__ == "__main__":
    main()