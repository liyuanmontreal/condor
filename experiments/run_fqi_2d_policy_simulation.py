import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.fqi_2d import FQI2D
from src.rl.condor_env_2d import Condor2DEnv
import numpy as np
import matplotlib.pyplot as plt


def extract_N(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(x[0])
    return float(x)


def main():
    # 1. 加载训练好的 FQI2D 模型
    fqi = FQI2D.load("outputs/fqi_release_lead.pkl")

    # 2. 初始化环境
    N0 = 566
    env = Condor2DEnv(N0=N0)

    years = list(range(2024, 2055))
    H = len(years)

    pops = []
    actions_release = []
    actions_mitig = []

    # 3. reset 环境
    state, _ = env.reset()

    # 4. 仿真
    for t in range(H):
        year = years[t]

        N_scalar = extract_N(state)
        u_t, e_t = fqi.select_action(N_scalar)

        actions_release.append(u_t)
        actions_mitig.append(e_t)

        next_state, reward, _, _, _ = env.step([u_t, e_t])
        pops.append(extract_N(next_state))

        state = next_state

    # 5. 打印结果
    print("\n[INFO] Population trajectory:", pops)
    print("\n[INFO] Actions (release, mitigation):")
    for y, u, e in zip(years, actions_release, actions_mitig):
        print(f"  Year {y}: release={u}, mitigation={e}")

    os.makedirs("outputs", exist_ok=True)

    # 6. 画种群曲线
    plt.figure(figsize=(12, 6))
    plt.plot(years, pops, "o--", label="FQI Population")
    plt.axhline(650, linestyle="--", color="orange", label="Target 650")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title("2D FQI Projection (Budget-constrained)")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/fig_rl_2d_target_projection.png", dpi=300)
    plt.show()

    # 7. 画放归动作曲线
    plt.figure(figsize=(12, 5))
    plt.plot(years, actions_release, "o-", color="blue", label="Release (u_t)")
    plt.xlabel("Year")
    plt.ylabel("Release count")
    plt.title("RL Policy: Release Actions Over Time")
    plt.grid()
    plt.legend()
    plt.savefig("outputs/fig_rl_2d_actions_release.png", dpi=300)
    plt.show()

    # 8. 画治理动作曲线
    plt.figure(figsize=(12, 5))
    plt.plot(years, actions_mitig, "o-", color="green", label="Mitigation (e_t)")
    plt.xlabel("Year")
    plt.ylabel("Lead mitigation effort")
    plt.title("RL Policy: Mitigation Actions Over Time")
    plt.grid()
    plt.legend()
    plt.savefig("outputs/fig_rl_2d_actions_mitig.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
