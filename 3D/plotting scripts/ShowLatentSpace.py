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

 
    vae = VariationalAutoencoder()
    vae.build(input_shape=(None, 32, 32 ,1))
    vae.encoder.summary()
    vae.decoder.summary()

    vae.load_weights("../saved_models/trained_weights_30").expect_partial()

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')


    for digit in range(0, 9 + 1):
        test_dataset = test_ds.apply(get_pipeline(digit))

        x_coords = []
        y_coords = []
        z_coords = []

        for imgs in test_dataset.take(10):
            embeddings, _, _ = vae.encoder(imgs)

            x_coords.append(embeddings[:, 0])
            y_coords.append(embeddings[:, 1])
            z_coords.append(embeddings[:, 2])

            
        ax.scatter(x_coords, y_coords, z_coords, label=digit)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")