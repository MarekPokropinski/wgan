import tensorflow as tf


class MnistGenerator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dim = 64
        self.input_to_2d = tf.keras.layers.Dense(4 * 4 * 4 * self.dim, activation="relu")
        self.block1 = tf.keras.layers.Conv2DTranspose(
            2 * self.dim, kernel_size=5, strides=2, padding="same", activation="relu"
        )
        self.blocks = [
            tf.keras.layers.Conv2DTranspose(self.dim, kernel_size=5, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh"),
        ]

    def call(self, x):
        x = self.input_to_2d(x)
        x = tf.reshape(x, shape=[-1, 4, 4, 4 * self.dim])
        x = self.block1(x)
        x = x[:, :7, :7, :]
        for block in self.blocks:
            x = block(x)
        return x


class MnistDiscriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dim = 64
        self.blocks = [
            tf.keras.layers.Conv2D(self.dim, kernel_size=5, strides=2, padding="same", activation="leaky_relu"),
            tf.keras.layers.Conv2D(2 * self.dim, kernel_size=5, strides=2, padding="same", activation="leaky_relu"),
            tf.keras.layers.Conv2D(4 * self.dim, kernel_size=5, strides=2, padding="same", activation="leaky_relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
        ]

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x
