import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Condor2DEnv(gym.Env):
    """
    2D Condor environment with:
      - release action u_t
      - lead mitigation action e_t (0/1)
    Dynamics calibrated so that: 
      * release=25 + mitigation=1  → slight positive growth (≈ +3 per year)
      * population approaches steady state near K=720
    """

    metadata = {"render_modes": []}

    def __init__(self, 
                 N0=566, 
                 K=720,
                 r=0.045,
                 base_mort=0.017,     # ↓ lowered from 0.02 (自然死亡略低)
                 lead_extra=0.009,    # ↓ lowered from 0.015 (铅额外死亡率略低)
                 lead_reduction=0.5,  # ↑ increased from 0.25 (治理可减少 50% 铅死亡)
                 alpha_release=0.40,  # ↓ release effectiveness tuned
                 max_release=40,
                 max_steps=50):

        super().__init__()

        # -------- ENV PARAMETERS --------
        self.N0 = float(N0)
        self.K = float(K)
        self.r = float(r)

        # -------- CALIBRATED PARAMETERS --------
        self.base_mort = float(base_mort)
        self.lead_extra = float(lead_extra)
        self.lead_reduction = float(lead_reduction)
        self.alpha_release = float(alpha_release)

        # -------- ACTION SPACE --------
        # action = [release, mitigation(0/1)]
        self.max_release = int(max_release)
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([max_release, 1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # -------- OBSERVATION --------
        # state = [population]
        self.observation_space = spaces.Box(
            low=np.array([0], dtype=np.float32),
            high=np.array([2000], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        self.max_steps = max_steps
        self.steps = 0
        self.state = np.array([self.N0], dtype=np.float32)

    # -------------------------------------------------------------
    def reset(self, *args, **kwargs):
        self.steps = 0
        self.state = np.array([self.N0], dtype=np.float32)
        return self.state, {}

    # -------------------------------------------------------------
    def step(self, action):
        release = float(np.clip(action[0], 0, self.max_release))
        mitigation = 1.0 if action[1] >= 0.5 else 0.0

        N = float(self.state[0])

        # ──────────────────────────────────────────────
        # LOGISTIC GROWTH
        growth = self.r * N * (1 - N / self.K)

        # NATURAL MORTALITY
        nat_mort = self.base_mort * N

        # LEAD MORTALITY (reduced if mitigation = 1)
        lead_effective = self.lead_extra * (1 - self.lead_reduction * mitigation)
        lead_mort = lead_effective * N

        # EFFECTIVE RECRUITMENT FROM RELEASE
        recruitment = self.alpha_release * release

        # ──────────────────────────────────────────────
        # NEW POPULATION
        N_next = N + growth - nat_mort - lead_mort + recruitment
        if N_next < 0:
            N_next = 0

        self.state = np.array([N_next], dtype=np.float32)
        self.steps += 1

        # Reward = move population toward target 650
        target = 650
        reward = -abs(N_next - target)

        done = self.steps >= self.max_steps

        return self.state, reward, done, False, {}

