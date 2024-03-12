import tensorflow as tf

from Encoder import *
from Decoder import *

class VariationalAutoencoder(tf.keras.Model):

    def __init__(self):
        super(VariationalAutoencoder, self).__init__()

        self.embedding_size = 3

        self.encoder = Encoder(self.embedding_size)
        self.decoder = Decoder()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.bce_loss = tf.keras.losses.MeanSquaredError()

        self.metric_total_loss = tf.keras.metrics.Mean(name="total_loss")
        self.metric_mse_loss = tf.keras.metrics.Mean(name="mse_loss")
        self.metric_kl_loss = tf.keras.metrics.Mean(name="kl_loss")

    @tf.function
    def call(self, x):
        
        embedding, mu, sigma = self.encoder(x)
        reconstructed_x = self.decoder(embedding)

        return reconstructed_x, mu, sigma

    @tf.function
    def get_kl_loss(self, mu, sigma):
        kl_loss = 0.5 * tf.math.reduce_sum(tf.square(mu) + tf.square(sigma) - 2*tf.math.log(sigma + 1e-10) - 1, axis=1)
        kl_loss = tf.reduce_mean(kl_loss)
        return kl_loss

    @tf.function
    def train_step(self, x):

        with tf.GradientTape() as tape:
            reconstructed_x, mu, sigma = self(x)
            mse_loss = self.bce_loss(x, reconstructed_x)
            kl_loss = self.get_kl_loss(mu, sigma)

            loss = mse_loss + 0.01 * kl_loss
    
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_total_loss.update_state(loss)
        self.metric_mse_loss.update_state(mse_loss)
        self.metric_kl_loss.update_state(kl_loss)

    def test_step(self, dataset):
          
        for x in dataset:
            reconstructed_x, mu, sigma = self(x)
            mse_loss = self.bce_loss(x, reconstructed_x)
            kl_loss = self.get_kl_loss(mu, sigma)

            loss = mse_loss + kl_loss

            self.metric_total_loss.update_state(loss)
            self.metric_mse_loss.update_state(mse_loss)
            self.metric_kl_loss.update_state(kl_loss)




