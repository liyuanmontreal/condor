import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    df = pd.read_csv("outputs/condor_net_action.csv")

    years = df["Year"].values
    net_action = df["NetAction"].values

    plt.figure(figsize=(10, 5))
    plt.plot(years, net_action, "o-", color="#0066cc", label="NetAction (u_t - D_t)")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.6)

    plt.xlabel("Year")
    plt.ylabel("NetAction (u_t - D_t)")
    plt.title("Latent NetAction Over Time")
    plt.grid(alpha=0.4)
    plt.legend()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/fig_net_action.png", dpi=300)
    plt.show()

    print("[INFO] Saved figure → outputs/fig_net_action.png")

if __name__ == "__main__":
    main()
