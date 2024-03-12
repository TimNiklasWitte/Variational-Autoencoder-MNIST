from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.lineplot(data=df.loc[:, "train total loss"], ax=axes[0], markers=True, label="train")
    sns.lineplot(data=df.loc[:, "test total loss"], ax=axes[0], markers=True, label="test")
    axes[0].set_title("Total loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid()

    sns.lineplot(data=df.loc[:, "train mse loss"], ax=axes[1], markers=True, label="train")
    sns.lineplot(data=df.loc[:, "test mse loss"], ax=axes[1], markers=True, label="test")
    axes[1].set_title("MSE loss")
    axes[1].set_ylabel("Loss")
    axes[1].grid()

    sns.lineplot(data=df.loc[:, "train kl loss"], ax=axes[2], markers=True, label="train")
    sns.lineplot(data=df.loc[:, "test kl loss"], ax=axes[2], markers=True, label="test")
    axes[2].set_title("KL loss")
    axes[2].set_ylabel("Loss")
    axes[2].grid()
    
    plt.tight_layout()
    plt.savefig("../plots/Loss.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")