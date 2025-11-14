import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.rl.fqi_2d import FQI2D


def main():
    df = pd.read_csv("outputs/rl_dataset_2d.csv")
    print(df.head())

    fqi = FQI2D()
    fqi.fit(df, iterations=30)
    fqi.save("outputs/fqi_2d.pkl")


if __name__ == "__main__":
    main()
