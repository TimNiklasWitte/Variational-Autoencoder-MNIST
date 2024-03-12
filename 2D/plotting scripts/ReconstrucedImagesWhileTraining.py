from LoadDataframe import *
from matplotlib import pyplot as plt

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)
    num_epochs = df.to_numpy().shape[0]

    batch_idx = 0
    for epoch in range(num_epochs):

        fig, axes = plt.subplots(nrows=1, ncols=2)

        img = df.loc[epoch, "images"][batch_idx]

        input = img[:, :32, :]
        axes[0].imshow(input)
        axes[0].set_title("Input")
        axes[0].axis("off")
        
        reconstructed = img[:, 32:, :]
        axes[1].imshow(reconstructed)
        axes[1].set_title("Reconstructed input")
        axes[1].axis("off")

        plt.suptitle(f"Epoch: {epoch}")
        plt.savefig(f"../plots/reconstructed images while training/epoch_{epoch}.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")