import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D


# def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
#     """
#     resample: None, 'down', or 'up'
#     """
#     if resample=='down':
#         conv_shortcut = tf.keras.layers.AveragePooling2D
#         conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
#         conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
#     elif resample=='up':
#         conv_shortcut = UpsampleConv
#         conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
#         conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
#     elif resample==None:
#         conv_shortcut = lib.ops.conv2d.Conv2D
#         conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
#         conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
#     else:
#         raise Exception('invalid resample value')

#     if output_dim==input_dim and resample==None:
#         shortcut = inputs # Identity skip-connection
#     else:
#         shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
#                                  he_init=False, biases=True, inputs=inputs)

#     output = inputs
#     output = Normalize(name+'.BN1', [0,2,3], output)
#     output = tf.nn.relu(output)
#     output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
#     output = Normalize(name+'.BN2', [0,2,3], output)
#     output = tf.nn.relu(output)
#     output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

#     return shortcut + output



class MnistGenerator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dim = 64
        self.input_to_2d = Dense(4*4*4*self.dim, activation="relu")
        self.block1 = tf.keras.layers.Conv2DTranspose(2*self.dim, kernel_size=5, strides=2, padding="same", activation="relu")
        self.blocks = [
            tf.keras.layers.Conv2DTranspose(self.dim, kernel_size=5, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="sigmoid"),
        ]

    def call(self, x):
        x = self.input_to_2d(x)
        x = tf.reshape(x, shape=[-1, 4, 4, 4*self.dim]) 
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
            tf.keras.layers.Conv2D(2*self.dim, kernel_size=5, strides=2, padding="same", activation="leaky_relu"),
            tf.keras.layers.Conv2D(4*self.dim, kernel_size=5, strides=2, padding="same", activation="leaky_relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ]

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x

