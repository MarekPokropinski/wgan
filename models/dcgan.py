import tensorflow as tf


class DCGanGenerator(tf.keras.Model):
    def __init__(self, z_dim, out_channels, upscaling_blocks=5):
        super().__init__()
        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
        self.num_out_channels = out_channels
        self.latent_size = z_dim
        self.upscaling_blocks = upscaling_blocks  # out size is 256x256 for upscaling_blocks=5
        self.base_size = 2 ** (self.upscaling_blocks - 1) * 32

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(4 * 4 * self.base_size, use_bias=False, input_shape=(self.latent_size,), kernel_initializer=self.weight_init))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Reshape((4, 4, self.base_size)))

        num_channels = self.base_size // 2
        for i in range(self.upscaling_blocks - 1):
            model.add(tf.keras.layers.Conv2DTranspose(num_channels, (5, 5), strides=(2, 2), padding="same", use_bias=False, kernel_initializer=self.weight_init))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())
            num_channels //= 2

        model.add(
            tf.keras.layers.Conv2DTranspose(
                self.num_out_channels,
                (5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
                kernel_initializer=self.weight_init
            )
        )
        return model

    def call(self, x):
        return self.model(x)


class DCGanDiscriminator(tf.keras.Model):
    def __init__(self, downscaling_blocks=5):
        super().__init__()
        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
        self.base_size = 32
        self.downscaling_blocks = downscaling_blocks
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()

        num_channels = self.base_size
        first_layer_kwargs = {}
        # first_layer_kwargs = {"input_shape": (128, 128, 3)}
        for i in range(self.downscaling_blocks):
            model.add(tf.keras.layers.Conv2D(num_channels, (5, 5), strides=(2, 2), padding="same", kernel_initializer=self.weight_init, **first_layer_kwargs))
            first_layer_kwargs = {}
            # model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LayerNormalization())
            model.add(tf.keras.layers.LeakyReLU())
            num_channels *= 2

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=self.weight_init))
        return model

    def call(self, x):
        return self.model(x)
