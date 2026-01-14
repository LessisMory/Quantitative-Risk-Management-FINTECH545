import numpy as np
import pandas as pd
from scipy import stats
from src.hw7_2 import simulate_t

def test_t_fitting(tmp_path):
    np.random.seed(42)
    data = np.random.standard_t(df = 5, size = 500)
    df = pd.DataFrame({"x":data})
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    df.to_csv(input_csv, index = False)
    simulate_t(input_csv, output_csv)
    output = pd.read_csv(output_csv)

    assert output.shape[1] == 3 and list(output.columns) == ["degree of freedom", "mu", "sigma"]
    assert len(output) == 1

    df = output.loc[0,"degree of freedom"]
    mu = output.loc[0,"mu"]
    sigma = output.loc[0,"sigma"]
    assert df > 2
    assert abs(mu - 0) < 0.2
    assert sigma > 0
