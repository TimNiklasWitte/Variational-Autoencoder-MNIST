import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import datetime

from VariationalAutoencoder import *

NUM_EPOCHS = 30
BATCH_SIZE = 32

def main():

    #
    # Load dataset
    #   
    train_ds, test_ds = tfds.load("mnist", split=["train", "test"], as_supervised=True)

    train_dataset = train_ds.apply(prepare_data)
    test_dataset = test_ds.apply(prepare_data)
    
    for x in test_dataset.take(1):
        x_tensorboard = x

    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Initialize model
    #
    vae = VariationalAutoencoder()
    vae.build(input_shape=(None, 32, 32 ,1))
    vae.encoder.summary()
    vae.decoder.summary()
 
    #
    # Train and test loss/accuracy
    #
    print(f"Epoch 0")
    log(train_summary_writer, vae, train_dataset, test_dataset, x_tensorboard, epoch=0)

    vae.save_weights(f"./saved_models/trained_weights_0", save_format="tf")

    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for x in tqdm.tqdm(train_dataset, position=0, leave=True): 
            vae.train_step(x)

        log(train_summary_writer, vae, train_dataset, test_dataset, x_tensorboard, epoch)

        # Save model (its parameters)
        vae.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")


def log(train_summary_writer, vae, train_dataset, test_dataset, x_tensorboard, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        vae.test_step(train_dataset.take(5000))

    #
    # Train
    #
    with train_summary_writer.as_default():
        for metric in vae.metrics:
            print(f"train_{metric.name}: {metric.result()}") 
            tf.summary.scalar(f"train_{metric.name}", metric.result(), step=epoch)
            metric.reset_states()

    #
    # Test
    #

    vae.test_step(test_dataset)

    with train_summary_writer.as_default():
        for metric in vae.metrics:
            print(f"test_{metric.name}: {metric.result()}") 
            tf.summary.scalar(f"test_{metric.name}", metric.result(), step=epoch)
            metric.reset_states()


    #
    # Write images to tensorboard
    # 
    x_reconstructed, _ ,_ = vae(x_tensorboard) # ignore mu, sigma

    imgs_tensorboard = tf.concat([x_tensorboard, x_reconstructed], axis=2)
    # [-1, 1] -> [0, 1]
    imgs_tensorboard = (imgs_tensorboard + 1)/2

    with train_summary_writer.as_default():
        tf.summary.image(name="images",data = imgs_tensorboard, step=epoch, max_outputs=BATCH_SIZE)


def prepare_data(dataset):

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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")