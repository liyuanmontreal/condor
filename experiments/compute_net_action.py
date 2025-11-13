import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import os

from src.utils import load_population_data, logistic_step
from src.rl.condor_env import CondorEnv


def compute_net_action(df, r=0.08, K=700.0, alpha=1.0):
    """
    根据种群变化 ΔN_t 和 logistic 增长 G_t 反推隐含动作：
        u_t - D_t = (ΔN_t - G_t) / alpha
    """
    years = df["Year"].values
    N = df["Total"].values

    net_actions = []
    for t in range(len(N) - 1):
        N_t = N[t]
        N_next = N[t + 1]

        delta = N_next - N_t
        G_t = r * N_t * (1 - N_t / K)

        net_u_minus_D = (delta - G_t) / alpha
        net_actions.append(net_u_minus_D)

    return pd.DataFrame({
        "Year": years[:-1],
        "NetAction": net_actions,
    })


def build_offline_dataset(df, net_df, r=0.08, K=700, alpha=1.0):
    """
    构建 Offline RL 数据集：
       state s_t = N_t
       action a_t = (u_t - D_t) 隐含动作
       reward r_t = -cost (来自 CondorEnv 奖励函数)
       next_state s_{t+1} = N_{t+1}
    """
    env = CondorEnv(r=r, K=K, alpha=alpha)

    states = df["Total"].values[:-1]
    next_states = df["Total"].values[1:]
    actions = net_df["NetAction"].values

    rewards = []
    for t in range(len(actions)):
        N_t = states[t]
        a_t = actions[t]

        # 使用环境公式计算奖励
        env.N = N_t
        env.t = t
        N_next, reward, done, info = env.step(a_t)
        rewards.append(reward)

    offline = pd.DataFrame({
        "Year": df["Year"].values[:-1],
        "State": states,
        "Action": actions,
        "Reward": rewards,
        "NextState": next_states,
    })
    return offline


def main():
    df = load_population_data("data/condor_population_strict.csv")

    os.makedirs("outputs", exist_ok=True)

    # 1) 计算隐含动作
    net_df = compute_net_action(df, r=0.08, K=700, alpha=1.0)
    net_df.to_csv("outputs/condor_net_action.csv", index=False)
    print("[INFO] Saved net actions → outputs/condor_net_action.csv")

    # 2) 构建 offline RL 数据集
    offline = build_offline_dataset(df, net_df)
    offline.to_csv("outputs/offline_dataset.csv", index=False)
    print("[INFO] Saved offline RL dataset → outputs/offline_dataset.csv")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
