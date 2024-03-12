import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self, embedding_size):
 
        super(Encoder, self).__init__()

        self.embedding_size = embedding_size

        self.layer_list = [
            # (batch_size, 32, 32, 1)
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 16, 16, 8)

            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 8, 8, 16)

            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 4, 4, 32)

            tf.keras.layers.Flatten(),
            # (batch_size, 512)
        ]

        self.mu_layer = tf.keras.layers.Dense(self.embedding_size, activation=None)
        self.log_sigma_layer = tf.keras.layers.Dense(self.embedding_size, activation=None)
    
    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)

        mu = self.mu_layer(x)
        log_sigma = self.log_sigma_layer(x)
        sigma = tf.exp(log_sigma / 2)

        # reparameterization trick
        batch_size = tf.shape(mu)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.embedding_size), mean=0, stddev=1)
    
        embedding = mu + (sigma * epsilon)

        return embedding, mu, sigma
    
    
