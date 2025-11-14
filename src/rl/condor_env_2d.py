import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Condor2DEnv(gym.Env):
    """
    2D action ecological model:
        action = (release, mitigation)

    release: 0–20 birds per year
    mitigation: 0 = no lead cleanup, 1 = cleanup applied
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, N0=566, start_year=2024):
        super().__init__()

        self.N = float(N0)
        self.year = start_year
        self.horizon_counter = 0
        self.H = 30  # simulation horizon

        # -------------------------
        # action space (2D)
        # -------------------------
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([20, 1], dtype=np.float32),
            dtype=np.float32
        )

        # -------------------------
        # observation space
        # -------------------------
        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([2000.0], dtype=np.float32),
            dtype=np.float32
        )

        # ecological parameters
        self.r = 0.06          # growth rate
        self.K = 700           # carrying capacity
        self.baseline_mort = 0.10
        self.lead_positive_rate = 0.87
        self.lead_specific_prop = 0.50

    # ------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.array([self.N], dtype=np.float32)
        return obs, {}

    # ------------------------------------------------
    def step(self, action):
        release = float(action[0])
        mitigation = float(action[1])

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

        # update population
        self.N = self.N + growth + release - mortality
        self.N = max(self.N, 0)

        # reward: closer to target population is better
        target = 650
        reward = -abs(self.N - target)

        self.year += 1
        self.horizon_counter += 1

        terminated = self.horizon_counter >= self.H

        obs = np.array([self.N], dtype=np.float32)
        return obs, reward, terminated, False, {}
