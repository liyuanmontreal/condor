import numpy as np
from sklearn.neural_network import MLPRegressor

class FQI2D:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=500,
            learning_rate_init=0.001
        )
        self.fitted = False

        # Discrete action grid
        self.release_levels = np.arange(0, 21, 1)
        self.mitigation_levels = np.array([0, 1])

    # -------------------------------------------
    def fit(self, df, iterations=25):
        X = df[["state_N", "release", "mitigation"]].values
        y = df["target_Q"].values

        for i in range(iterations):
            print(f"[FQI] iter {i+1}/{iterations}")
            self.model.fit(X, y)
            y = self.model.predict(X)  # update target Q

        self.fitted = True
        print("[FQI] Training completed.")

    # -------------------------------------------
    def select_action(self, state_N):
        best_q = -1e18
        best_pair = (0, 0)

        for u in self.release_levels:
            for e in self.mitigation_levels:
                q = self.model.predict([[state_N, u, e]])[0]
                if q > best_q:
                    best_q = q
                    best_pair = (u, e)

        return best_pair
