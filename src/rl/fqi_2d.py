# src/rl/fqi_2d.py

import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

class FQI2D:
    def __init__(self,
                 release_levels=[0, 5, 10, 15, 20, 25, 30],
                 mitigation_levels=[0, 1],
                 gamma=0.99):
        self.release_levels = release_levels
        self.mitigation_levels = mitigation_levels
        self.gamma = gamma

        # Q-function approximator
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            max_iter=500,
            random_state=0
        )

        self.fitted = False

    def fit(self, df, iterations=30):
        s = df["state_N"].values.astype(float)
        u = df["release"].values.astype(float)
        e = df["mitigation"].values.astype(float)
        r = df["reward"].values.astype(float)
        s2 = df["next_N"].values.astype(float)

        # Inputs: (s, u, e)
        X = np.column_stack([s, u, e])

        # Q targets
        y = r.copy()

        print(f"[INFO] FQI training: {len(df)} samples")

        for it in range(iterations):
            print(f"[FQI] iteration {it+1}/{iterations}")

            # ---- iteration 1: Q(s’) = 0 ----
            if it == 0:
                y_target = r.copy()
            else:
                # estimate Q(s', a')
                q_next = []
                for i in range(len(s2)):
                    best_q = -1e9
                    for u2 in self.release_levels:
                        for e2 in self.mitigation_levels:
                            q_val = self.model.predict([[s2[i], u2, e2]])[0]
                            best_q = max(best_q, q_val)
                    q_next.append(best_q)

                q_next = np.array(q_next)
                y_target = r + self.gamma * q_next

            # fit model
            self.model.fit(X, y_target)
            self.fitted = True

        print("[INFO] FQI training completed!")

    def save(self, path="outputs/fqi_release_lead.pkl"):
        joblib.dump(self, path)
        print(f"[INFO] Saved FQI model → {path}")

    @staticmethod
    def load(path="outputs/fqi_release_lead.pkl"):
        return joblib.load(path)

    def select_action(self, state):
        """Returns tuple: (release, mitigation)"""
        if isinstance(state, (tuple, list, np.ndarray)):
            state_N = float(state[0])
        else:
            state_N = float(state)

        best_q = -1e9
        best_pair = (0, 0)

        for u in self.release_levels:
            for e in self.mitigation_levels:
                q = self.model.predict([[state_N, u, e]])[0]
                if q > best_q:
                    best_q = q
                    best_pair = (u, e)

        return best_pair
