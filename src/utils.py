import pandas as pd
import numpy as np


def load_population_data(path):
    """
    Load population CSV and clean formatting issues.
    Expect columns: Year, Total
    """
    df = pd.read_csv(path)

    # 修理可能的字符串问题（如 "# 1992"）
    df["Year"] = (
        df["Year"]
        .astype(str)
        .str.replace("#", "")
        .str.replace(" ", "")
        .astype(int)
    )

    df["Total"] = df["Total"].astype(float)

    return df.sort_values("Year").reset_index(drop=True)


def logistic_step(N, action, r=0.08, K=700.0, alpha=1.0):
    """
    Single-step logistic growth with action u_t - D_t:
        N_{t+1} = N + rN(1 - N/K) + alpha * action

    Parameters:
        N      : current population
        action : net action (u_t - D_t)
        r      : intrinsic growth rate
        K      : carrying capacity
        alpha  : action impact coefficient
    """
    growth = r * N * (1 - N / K)
    N_next = N + growth + alpha * action

    # 保证生态合理性：不能为负
    N_next = max(N_next, 0.0)

    return N_next
