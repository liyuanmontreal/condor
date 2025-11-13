import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


class FittedQIteration:
    """
    Minimal offline RL: fitted Q-iteration for 1D population control.
    """

    def __init__(self, gamma=0.95):
        self.gamma = gamma
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 32),
            max_iter=500,
            learning_rate_init=1e-3
        )

    def fit(self, df, iterations=30):
        """
        df: offline_dataset.csv with columns:
            ['State', 'Action', 'Reward', 'NextState']
        """
        s = df["State"].values
        a = df["Action"].values
        r = df["Reward"].values
        s_next = df["NextState"].values

        X = np.vstack([s, a]).T

        # initialize Q(s,a) ~ reward-only
        y = r.copy()

        for it in range(iterations):
            self.model.fit(X, y)

            # compute Q targets
            Q_next = []
            for i in range(len(df)):
                # evaluate for candidate action a' = {-20..20}
                a_candidates = np.linspace(-20, 20, 41)
                q_vals = [self.model.predict([[s_next[i], a]])[0]
                          for a in a_candidates]
                Q_next.append(max(q_vals))

            y = r + self.gamma * np.array(Q_next)
            print(f"[FQI] iter {it+1}/{iterations}")

        print("[FQI] Training completed.")

    def select_action(self, state):
        """
        Choose best a in candidate range.
        """
        a_candidates = np.linspace(-20, 20, 41)
        qs = [self.model.predict([[state, a]])[0] for a in a_candidates]
        return a_candidates[np.argmax(qs)]
