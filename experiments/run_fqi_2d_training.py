import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import joblib
from src.rl.fqi_2d import FQI2D

def main():
    df = pd.read_csv("outputs/rl_dataset_2d.csv")
    print(f"[INFO] Loaded offline dataset with {len(df)} transitions.")

    fqi = FQI2D()
    fqi.fit(df, iterations=25)

    joblib.dump(fqi, "outputs/fqi_release_lead.pkl")
    print("[INFO] Saved model to outputs/fqi_release_lead.pkl")

if __name__ == "__main__":
    main()
