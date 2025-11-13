import numpy as np
from src.utils import logistic_step


class CondorEnv:
    """
    Minimal environment for Condor population control.

    State:      N_t (population)
    Action:     a_t = net action (u_t - D_t)
    Dynamics:   N_{t+1} = logistic_step(...)
    Reward:     negative error from target
    """

    def __init__(self, r=0.08, K=700.0, alpha=1.0):
        self.r = r
        self.K = K
        self.alpha = alpha

        self.N = None
        self.t = 0

        # dynamic recovery target:
        # 566 -> 600 (recovery) -> 700 (long-term)
        self.target_mid = 600
        self.target_final = 700

        # projection horizon to compute dynamic target
        self.T_transition = 30  # years to move from 600 → 700

    def reset(self, N0=None):
        if N0 is None:
            self.N = 566.0  # start from real population
        else:
            self.N = float(N0)

        self.t = 0
        return np.array([self.N], dtype=np.float32)

    def compute_target(self):
        """
        A smooth target function:
            600 at t=0
            700 at t=T_transition
        """
        progress = min(self.t / self.T_transition, 1.0)
        return self.target_mid + progress * (self.target_final - self.target_mid)

    def step(self, action):
        N_t = float(self.N)

        # population update
        N_next = logistic_step(N_t, action, r=self.r, K=self.K, alpha=self.alpha)

        # compute reward
        target = self.compute_target()
        reward = -abs(N_next - target)

        # update
        self.N = N_next
        self.t += 1

        done = False  # episodic horizon is outside this minimal version
        info = {"target": target}

        return (
            np.array([N_next], dtype=np.float32),
            reward,
            done,
            info
        )
