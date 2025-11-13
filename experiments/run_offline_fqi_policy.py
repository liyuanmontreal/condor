import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.rl.offline_fqi import FittedQIteration
from src.utils import logistic_step

def main():
    df = pd.read_csv("outputs/offline_dataset.csv")

    # 1) Train FQI offline RL  //use net action to train RL
    fqi = FittedQIteration(gamma=0.95)
    fqi.fit(df, iterations=25)

    # 2) Predict trajectory for 30 years
    N0 = df["State"].values[-1]  # last known real value
    T = 30

    traj = [N0]
    actions = []

    state = N0
    for t in range(T):
        a_t = fqi.select_action(state)
        actions.append(a_t)

        next_state = logistic_step(state, a_t, r=0.08, K=700)
        traj.append(next_state)
        state = next_state

    # years_future = np.arange(df["Year"].values[-1] + 1,
    #                          df["Year"].values[-1] + 1 + T + 1)
    # years should have same length as actions & traj[1:]
    years_future = np.arange(
        df["Year"].values[-1] + 1,
        df["Year"].values[-1] + 1 + T
    )


    # 3) Save
    os.makedirs("outputs", exist_ok=True)

    # pd.DataFrame({
    #     "Year": years_future,
    #     "Population": traj[1:],
    #     "Action": actions
    # }).to_csv("outputs/fqi_policy_projection.csv", index=False)

    pop_future = traj[1:]       # length T
    actions_future = actions    # length T

    df_out = pd.DataFrame({
        "Year": years_future,
        "Population": pop_future,
        "Action": actions_future
    })

    df_out.to_csv("outputs/fqi_policy_projection.csv", index=False)


    # 4) Plot
    plt.figure(figsize=(10,6))
    plt.plot(years_future, traj[1:], "o--", color="darkorange", label="FQI Predicted")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("Offline RL (Fitted-Q) Projection")
    plt.grid(alpha=0.4)
    plt.legend()

    plt.savefig("outputs/fig_fqi_projection.png", dpi=300)
    plt.show()

    print("[INFO] Saved → outputs/fqi_policy_projection.csv")
    print("[INFO] Saved → outputs/fig_fqi_projection.png")

if __name__ == "__main__":
    main()
