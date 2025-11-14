
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from src.rl.condor_env_2d import Condor2DEnv

def simulate_one_step(N, release, mitigation):
    env = Condor2DEnv(N0=N)
    obs, info = env.reset()
    obs2, reward, term, trunc, info = env.step([release, mitigation])
    next_N = float(obs2[0])
    return reward, next_N

def main():
    df = pd.read_csv("data/condor_population_strict.csv")
    df = df[df["Year"] >= 2011]

    N_list = df["Total"].values.astype(float)

    releases = np.random.randint(0, 20, size=len(N_list)-1)
    mitigations = np.random.randint(0, 2, size=len(N_list)-1)

    data = []
    gamma = 0.99

    for t in range(len(releases)-1):
        N = N_list[t]
        u = releases[t]
        e = mitigations[t]

        reward, nextN = simulate_one_step(N, u, e)
        target_Q = reward + gamma * abs(nextN - 650) * -1

        data.append([N, u, e, target_Q])

    df2 = pd.DataFrame(data, columns=["state_N", "release", "mitigation", "target_Q"])
    df2.to_csv("outputs/rl_dataset_2d.csv", index=False)

    print("[INFO] Dataset saved to outputs/rl_dataset_2d.csv")

if __name__ == "__main__":
    main()
