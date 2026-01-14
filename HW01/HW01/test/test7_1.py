from src.hw7_1 import simulate_normal
import numpy as np
import pandas as pd
from scipy import stats

def test_normal_fitting(tmp_path):
    np.random.seed(0)
    data = np.random.normal(loc = 1.0, scale = 3.0, size = 500)
    df = pd.DataFrame({"x":data})
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    df.to_csv(input_csv, index=False)
    simulate_normal(input_csv, output_csv)
    output = pd.read_csv(output_csv)

    assert output.shape[1] == 3 and list(output.columns) == ["mu", "sigma", "unbiased sigma"]
    assert len(output) == 1
    mu_hat = output.loc[0,"mu"]
    sigma_hat = output.loc[0,"sigma"]
    sigma_adj_hat = output.loc[0,"unbiased sigma"]

    assert abs(mu_hat - 1.0) < 0.2
    assert abs(sigma_adj_hat - 3.0) < 0.2
    assert sigma_adj_hat > sigma_hat