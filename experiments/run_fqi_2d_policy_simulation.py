
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import joblib
from src.rl.condor_env_2d import Condor2DEnv

def extract_N(state):
    """Robust extractor for scalar N."""
    if isinstance(state, tuple):
        return extract_N(state[0])
    if isinstance(state, dict):
        return extract_N(next(iter(state.values())))
    if isinstance(state, np.ndarray):
        return float(state.flatten()[0])
    if isinstance(state, (list, tuple)):
        return extract_N(state[0])
    return float(state)

def main():
    print("[INFO] Loaded FQI policy.")
    fqi = joblib.load("outputs/fqi_release_lead.pkl")

    start_year = 2024
    env = Condor2DEnv(N0=566, start_year=start_year)
    print("[INFO] Using N0 = 566 for start.")

    state, _ = env.reset()
    state = extract_N(state)

    populations = [state]
    releases, mitigations = [], []

    for t in range(30):
        u, e = fqi.select_action(state)
        releases.append(u)
        mitigations.append(e)

        result = env.step([u, e])
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result

        state = extract_N(obs)
        populations.append(state)

    years = list(range(start_year, start_year + len(populations)))

    plt.figure(figsize=(10,6))
    plt.plot(years, populations, "o--", label="FQI Population", color="#0055cc")
    plt.axhline(650, linestyle="--", color="gray", label="Target 650")
    plt.title("2D FQI Projection (Release + Lead Mitigation)")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("outputs/fqi_2d_projection.png")
    plt.show()

    print("[INFO] Saved outputs/fqi_2d_projection.png")

if __name__ == "__main__":
    main()
