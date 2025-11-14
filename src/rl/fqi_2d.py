# src/rl/fqi_2d.py
# fix budgt version
import numpy as np
import joblib
from itertools import product
from sklearn.neural_network import MLPRegressor


class FQI2D:
    """
    简单 Fitted Q-Iteration，用于 1D 状态 + 2D 动作（release, mitigation）

    数据格式要求 DataFrame 列：
      - state_N
      - release
      - mitigation
      - reward
      - next_N
    """

    def __init__(
        self,
        release_levels=None,
        mitigation_levels=None,
        hidden_layer_sizes=(64, 64),
        gamma=0.97,
        random_state=0,
    ):
        if release_levels is None:
            release_levels = [15, 20, 25]
        if mitigation_levels is None:
            mitigation_levels = [0, 1]

        self.release_levels = list(release_levels)
        self.mitigation_levels = list(mitigation_levels)
        self.gamma = gamma

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
        )

    # ---------- 内部辅助：构造 (state, action) 批 ----------
    def _sa_batch(self, states, releases, mitigations):
        """
        states: shape (N,)
        releases, mitigations: 标量 or shape (N,)
        返回形状 (N, 3) 的特征 [s, u, e]
        """
        s = np.asarray(states).reshape(-1, 1)
        u = np.asarray(releases).reshape(-1, 1)
        e = np.asarray(mitigations).reshape(-1, 1)
        X = np.concatenate([s, u, e], axis=1)
        return X

    # ---------- 训练 ----------
    def fit(self, df, iterations=30):
        """
        df: pandas.DataFrame，包含 state_N, release, mitigation, reward, next_N
        """
        s = df["state_N"].values.astype(float)
        u = df["release"].values.astype(float)
        e = df["mitigation"].values.astype(float)
        r = df["reward"].values.astype(float)
        s2 = df["next_N"].values.astype(float)

        # 初始 Q 目标：直接回归 reward
        X = self._sa_batch(s, u, e)
        y = r.copy()

        for it in range(iterations):
            # 拟合当前 Q
            self.model.fit(X, y)

            # 计算下一个状态的 max_a' Q(s', a')
            q_next_max = []
            for s_next in s2:
                q_values = []
                for u2, e2 in product(self.release_levels, self.mitigation_levels):
                    X_next = self._sa_batch([s_next], [u2], [e2])
                    q_val = self.model.predict(X_next)[0]
                    q_values.append(q_val)
                q_next_max.append(max(q_values))
            q_next_max = np.array(q_next_max)

            # 更新 target
            y = r + self.gamma * q_next_max

            print(f"[FQI] iter {it+1}/{iterations}")

        print("[FQI] Training completed.")

    # ---------- 策略：给定状态选动作 ----------
    def select_action(self, state_scalar):
        """
        输入：state_scalar（float or int）
        输出： (release, mitigation)
        """
        s_val = float(state_scalar)
        q_values = []
        actions = []
        for u, e in product(self.release_levels, self.mitigation_levels):
            X = self._sa_batch([s_val], [u], [e])
            q = self.model.predict(X)[0]
            q_values.append(q)
            actions.append((u, e))
        best_idx = int(np.argmax(q_values))
        return actions[best_idx]

    # ---------- 保存 / 加载 ----------
    def save(self, path: str):
        joblib.dump(self, path)
        print(f"[FQI] Saved model to {path}")

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        if not isinstance(obj, FQI2D):
            raise TypeError(f"Loaded object from {path} is not FQI2D")
        print(f"[FQI] Loaded model from {path}")
        return obj
