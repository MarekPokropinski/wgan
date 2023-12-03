from collections import defaultdict
from datetime import datetime
import tensorflow as tf
import tqdm
import numpy as np
from pathlib import Path

from wgan import WGAN
from models.mnist_models import MnistDiscriminator, MnistGenerator

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

log_dir = "logs/mnist/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/train"
train_writer = tf.summary.create_file_writer(log_dir)
train_writer.set_as_default()


def get_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    ds = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .map(lambda x: tf.cast(x[..., tf.newaxis], tf.float32) / 127.5 - 1.0)
        .cache()
        .shuffle(x_train.shape[0])
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


batch_size = 64
z_dim = 128
z_vis = tf.random.normal((16, z_dim))

if __name__ == "__main__":
    generator = MnistGenerator()
    discriminator = MnistDiscriminator()

    gan = WGAN(generator, discriminator, z_dim=z_dim, lr=1e-4)
    ds = get_dataset(batch_size=batch_size)

    pred = gan.generator(z_vis)
    print("generated shape:", pred.shape)

    generator_losses = []
    discriminator_losses = []

    progress_bar = tqdm.trange(20000)
    for iteration in progress_bar:
        for critic_step, real_data in enumerate(ds):
            discriminator_metrics = gan.discriminator_train_step(real_data)
            for metric_name, metric_value in discriminator_metrics.items():
                tf.summary.scalar(metric_name, data=metric_value.numpy(), step=critic_step)
            discriminator_losses.append(discriminator_metrics["discriminator_loss"].numpy())

            if critic_step % 5 == 4:
                generator_loss = gan.generator_train_step(batch_size)
                generator_losses.append(generator_loss.numpy())
                tf.summary.scalar("generator_loss", data=generator_loss, step=critic_step)

            desc = f"generator loss: {np.mean(generator_losses[-100:]):05f} discriminator loss: {np.mean(discriminator_losses[-100:]):05f}"
            progress_bar.set_description(desc)
            if critic_step % 100 == 0:
                gan.generator.save("saved_models/mnist/generator.keras")
                gan.discriminator.save("saved_models/mnist/discriminator.keras")

                generated = gan.generator(z_vis)
                tf.summary.image("Generated", (generated + 1) / 2, max_outputs=16, step=critic_step)
                tf.summary.image("Real", (real_data + 1) / 2, max_outputs=16, step=critic_step)
