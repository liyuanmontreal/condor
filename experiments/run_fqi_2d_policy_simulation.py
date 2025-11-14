
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from src.rl.fqi_2d import FQI2D
from src.rl.condor_env_2d import Condor2DEnv


def extract_N(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[0])
    return float(x)


def main():
    fqi = FQI2D.load("outputs/fqi_2d.pkl")

    # simulate from 566
    N0 = 566
    env = Condor2DEnv(N0=N0)

    years = list(range(2024, 2055))
    pops = []
    actions = []

    state, _ = env.reset()

    for t in range(len(years)):
        state_scalar = extract_N(state)
        u, e = fqi.select_action(state_scalar)
        actions.append((u, e))

        next_state, reward, _, _, _ = env.step([u, e])
        pops.append(extract_N(next_state))
        state = next_state

    print("[INFO] Population trajectory:")
    print(pops)

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(years, pops, "o--", label="FQI Population")
    plt.plot(years, [650] * len(years), "--", label="Target 650")

    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("2D FQI Projection")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/fig_rl_2d_target_projection.png", dpi=300)
    plt.show()
 


if __name__ == "__main__":
    main()
