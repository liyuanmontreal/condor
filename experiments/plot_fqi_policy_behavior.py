import numpy as np
import matplotlib.pyplot as plt
from src.rl.offline_fqi import FittedQIteration
import pandas as pd
import os

def main():
    df = pd.read_csv("outputs/offline_dataset.csv")

    fqi = FittedQIteration()
    fqi.fit(df, iterations=25)

    populations = np.linspace(0, 800, 200)
    actions = [fqi.select_action(N) for N in populations]

    plt.figure(figsize=(10,5))
    plt.plot(populations, actions, "-", color="#0099cc")
    plt.xlabel("Population")
    plt.ylabel("Optimal Action (Net Release - Death Mitigation)")
    plt.title("Offline RL Learned Policy")
    plt.grid(alpha=0.4)

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/fig_fqi_policy_behavior.png", dpi=300)
    plt.show()

    print("[INFO] Saved policy → outputs/fig_fqi_policy_behavior.png")

if __name__ == "__main__":
    main()
