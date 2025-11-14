
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.rl.condor_env_2d import Condor2DEnv


def main():
    df_pop = pd.read_csv("data/condor_population_strict.csv")  # must contain Year, Total
    df_rel = pd.read_csv("data/condor_releases.csv")    # Year, Release
    df_lead = pd.read_csv("data/lead_mitigation_proxy.csv")  # Year, MitigationLevel(0/1)

    # merge
    df = df_pop.merge(df_rel, on="Year", how="left")
    df = df.merge(df_lead, on="Year", how="left")

    # fill missing: assume 0 release, mitigation=0 pre-2011
    df["Release"] = df["Release"].fillna(0)
    df["Mitigation"] = df["Mitigation"].fillna(0)

    # only use 2011-2024 for offline RL
    df = df[df["Year"] >= 2011].reset_index(drop=True)

    records = []
    env = Condor2DEnv()

    for i in range(len(df) - 1):
        N_t = df.loc[i, "Total"]
        u_t = df.loc[i, "Release"]
        e_t = df.loc[i, "Mitigation"]
        N_tp1 = df.loc[i + 1, "Total"]

        reward = -abs(N_tp1 - 650) / 50.0
        reward -= 0.01 * u_t
        reward += 0.2 * e_t

        records.append({
            "state_N": N_t,
            "release": u_t,
            "mitigation": e_t,
            "reward": reward,
            "next_N": N_tp1
        })

    pd.DataFrame(records).to_csv("outputs/rl_dataset_2d.csv", index=False)
    print("[INFO] Saved dataset to outputs/rl_dataset_2d.csv")


if __name__ == "__main__":
    main()
