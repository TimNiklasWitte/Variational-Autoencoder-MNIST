import sys
sys.path.append("../")

import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

from VariationalAutoencoder import *
from Training import *

def get_pipeline(digit):

    def prepare_data(dataset):

        dataset = dataset.filter(lambda img, label: label == digit) # only digits
        
        # Remove label
        dataset = dataset.map(lambda img, target: img)

        # Convert data from uint8 to float32
        dataset = dataset.map(lambda img: tf.cast(img, tf.float32) )

        # Normalization: [0, 255] -> [-1, 1]
        dataset = dataset.map(lambda img: (img/128.)-1. )

        # Resize 28x28 -> 32x32
        dataset = dataset.map(lambda img: tf.image.resize(img, size=[32,32]) )

        # Cache
        dataset = dataset.cache()
        
        #
        # Shuffle, batch, prefetch
        #
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
    
    return prepare_data
    


def main():

    test_ds = tfds.load("mnist", split="test", as_supervised=True)

    for epoch in range(0,31):
        vae = VariationalAutoencoder()
        vae.build(input_shape=(None, 32, 32 ,1))
        vae.encoder.summary()
        vae.decoder.summary()

        vae.load_weights(f"../saved_models/trained_weights_{epoch}").expect_partial()

        for digit in range(0, 9 + 1):
            test_dataset = test_ds.apply(get_pipeline(digit))

            x_coords = []
            y_coords = []
    
            for imgs in test_dataset.take(10):
                embeddings, _, _ = vae.encoder(imgs)

                x_coords.append(embeddings[:, 0])
                y_coords.append(embeddings[:, 1])


            plt.scatter(x_coords, y_coords, label=digit, s=5)

        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
    

        plt.tight_layout()
        plt.legend(loc="lower right")
        plt.savefig(f"../plots/latent space/epoch_{epoch}.png", bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")