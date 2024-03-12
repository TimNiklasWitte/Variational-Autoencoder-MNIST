import tensorflow as tf

class Decoder(tf.keras.Model):

    def __init__(self):
 
        super(Decoder, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(4 * 4 * 32, activation='relu'),
            # (batch_size, 512) 4*4*32 = 512

            tf.keras.layers.Reshape((4, 4, 32)), 
            # (batch_size, 4, 4, 32)

            tf.keras.layers.Conv2DTranspose(16, kernel_size=(3,3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 8, 8, 16)

            tf.keras.layers.Conv2DTranspose(8, kernel_size=(3,3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 16, 16, 8)

            tf.keras.layers.Conv2DTranspose(1, kernel_size=(3,3), strides=(2,2), padding='same', activation='tanh'),
            # (batch_size, 32, 32, 1)
        ]

    
    @tf.function
    def call(self, x):
     
        for layer in self.layer_list:
            x = layer(x)
            
        return x
    
