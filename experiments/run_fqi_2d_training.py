import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.rl.fqi_2d import FQI2D
from src.rl.condor_env_2d import Condor2DEnv


def main():
    # 1. load offline data
    data_path = "outputs/rl_dataset_2d.csv"
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded offline dataset from {data_path} with {len(df)} transitions.")
    print(df.head())

    
    fqi = FQI2D(
        release_levels=[15, 20, 25],   # release bird number limit to {15, 20, 25} x {0, 1}
        mitigation_levels=[0, 1],
        hidden_layer_sizes=(64, 64),
        gamma=0.97,
        random_state=0,
    )

    # 3. train
    fqi.fit(df, iterations=30)

    # 4. save model
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/fqi_release_lead.pkl"
    fqi.save(model_path)


if __name__ == "__main__":
    main()

