from functools import partial
import tensorflow as tf


class ConvMeanPool(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(*args, **kwargs)
        self.mean_pool = tf.keras.layers.AvgPool2D()

    def call(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.mean_pool(x)
        return x


class MeanPoolConv(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(*args, **kwargs)
        self.mean_pool = tf.keras.layers.AvgPool2D()

    def call(self, inputs):
        x = inputs
        x = self.mean_pool(x)
        x = self.conv(x)
        return x


class UpscaleConv(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(*args, **kwargs)
        self.upscale = tf.keras.layers.UpSampling2D()

    def call(self, inputs):
        x = inputs
        x = self.upscale(x)
        x = self.conv(x)
        return x


class ResidualBlock(tf.keras.Model):
    normalizations = {
        "batch": tf.keras.layers.BatchNormalization,
        "layer": tf.keras.layers.LayerNormalization,
    }

    def __init__(self, output_dims, resample=None, normalization="batch"):
        super().__init__()
        self.resample = resample
        self.output_dims = output_dims
        normalization_type = self.normalizations[normalization]
        self.norm1 = normalization_type()
        self.norm2 = normalization_type()

    def build(self, input_shape):
        input_dim = input_shape[-1]
        output_shape = (*input_shape[:-1], self.output_dims)
        conv_1_kwargs = dict(kernel_size=3, use_bias=False, kernel_initializer="he_uniform", padding="same")
        conv_2_kwargs = dict(kernel_size=3, use_bias=True, kernel_initializer="he_uniform", padding="same")

        if self.resample == None:
            shortcut_type = tf.keras.layers.Conv2D
            self.conv1 = tf.keras.layers.Conv2D(filters=input_dim, **conv_1_kwargs)
            self.conv2 = tf.keras.layers.Conv2D(filters=self.output_dims, **conv_2_kwargs)

            self.conv1.build(input_shape)
            self.conv2.build(input_shape)

            self.norm1.build(input_shape)
            self.norm2.build(input_shape)
        elif self.resample == "up":
            shortcut_type = UpscaleConv
            self.conv1 = UpscaleConv(filters=self.output_dims, **conv_1_kwargs)
            self.conv2 = tf.keras.layers.Conv2D(filters=self.output_dims, **conv_2_kwargs)

            self.conv1.build(input_shape)
            self.conv2.build(output_shape)

            self.norm1.build(input_shape)
            self.norm2.build(output_shape)
        elif self.resample == "down":
            shortcut_type = MeanPoolConv
            self.conv1 = tf.keras.layers.Conv2D(filters=input_dim, **conv_1_kwargs)
            self.conv2 = ConvMeanPool(filters=self.output_dims, **conv_2_kwargs)

            self.conv1.build(input_shape)
            self.conv2.build(input_shape)

            self.norm1.build(input_shape)
            self.norm2.build(input_shape)
        if self.output_dims == input_dim and self.resample == None:
            self.shortcut = lambda x: x
        else:
            self.shortcut = shortcut_type(filters=self.output_dims, kernel_size=1, use_bias=True)
            self.shortcut.build(input_shape)

    def call(self, inputs):
        shortcut = self.shortcut(inputs)
        x = inputs
        x = self.norm1(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        return x + shortcut


def Generator(dim=64, out_channels=3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4 * 4 * 8 * dim))
    model.add(tf.keras.layers.Reshape([4, 4, 8 * dim]))

    model.add(ResidualBlock(8 * dim, resample="up", normalization="batch"))
    model.add(ResidualBlock(4 * dim, resample="up", normalization="batch"))
    model.add(ResidualBlock(2 * dim, resample="up", normalization="batch"))
    model.add(ResidualBlock(1 * dim, resample="up", normalization="batch"))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(out_channels, kernel_size=3, activation="tanh", padding="same"))
    return model


def Discriminator(dim=64):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=dim, kernel_size=3, padding="same"))

    model.add(ResidualBlock(2 * dim, resample="down", normalization="layer"))
    model.add(ResidualBlock(4 * dim, resample="down", normalization="layer"))
    model.add(ResidualBlock(8 * dim, resample="down", normalization="layer"))
    model.add(ResidualBlock(8 * dim, resample="down", normalization="layer"))

    model.add(tf.keras.layers.Reshape([4 * 4 * 8 * dim]))

    model.add(tf.keras.layers.Dense(1))
    return model
