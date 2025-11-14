import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Condor2DEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, N0=566):
        super().__init__()

        self.N = float(N0)
        self.year = 2024
        self.H = 30
        self.step_count = 0

        # updated ecological parameters
        self.r = 0.08              # stronger growth
        self.K = 700               # carrying capacity

        self.baseline_mort = 0.02  # 2% annual natural mortality
        self.lead_positive_rate = 0.30
        self.lead_specific_prop = 0.20  # realistic 0.12% mortality

        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([20, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([2000.0], dtype=np.float32),
            dtype=np.float32
        )

    # ---------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([self.N], dtype=np.float32), {}

    # ---------------------------
    def step(self, action):
        release, mitigation = float(action[0]), float(action[1])

        # logistic growth
        growth = self.r * self.N * (1 - self.N / self.K)

        # lead mortality
        lead_mortality_rate = (
            self.baseline_mort *
            self.lead_positive_rate *
            self.lead_specific_prop *
            (1 - mitigation)
        )
        mortality = lead_mortality_rate * self.N

        self.N = self.N + growth + release - mortality
        self.N = max(self.N, 0)

        # reward = target tracking + action reward + emergency rule
        target = 650
        reward = (
            -abs(self.N - target)
            + 20 * release
            + 40 * mitigation
        )

        # emergency boost: small population must recover
        if self.N < 550:
            reward += 200

        self.step_count += 1
        terminated = self.step_count >= self.H

        return np.array([self.N], dtype=np.float32), reward, terminated, False, {}
