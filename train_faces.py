from collections import defaultdict
from datetime import datetime
import tensorflow as tf
import tqdm
import numpy as np
from pathlib import Path
from models.good_models import Discriminator, Generator

from wgan import WGAN

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

log_dir = "logs/faces/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/train"
train_writer = tf.summary.create_file_writer(log_dir)
train_writer.set_as_default()

batch_size = 128
path = Path.home() / "portraits"
dataset_kwargs = dict(directory=path, labels=None, image_size=(256, 256), batch_size=batch_size, validation_split=0.05)
target_size = (64, 64)
seed = 364065600


def get_dataset():
    ds = tf.keras.utils.image_dataset_from_directory(**dataset_kwargs, subset="training", seed=seed)
    ds = ds.map(lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0).map(ImageAugmentations()).prefetch(tf.data.AUTOTUNE)
    return ds


def get_validation_dataset():
    ds = tf.keras.utils.image_dataset_from_directory(**dataset_kwargs, subset="validation", seed=seed)
    ds = (
        ds.map(lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0)
        .map(lambda x: tf.image.resize(x, size=target_size))
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


class ImageAugmentations:
    def __init__(self) -> None:
        self.RandomFlip = tf.keras.layers.RandomFlip(mode="horizontal")
        self.RandomRotation = tf.keras.layers.RandomRotation(0.1)
        self.RandomTranslation = tf.keras.layers.RandomTranslation(0.03, 0.03)
        self.RandomZoom = tf.keras.layers.RandomZoom(0.05)

    def __call__(self, img):
        x = img
        x = self.RandomFlip(x)
        x = self.RandomTranslation(x)
        x = self.RandomZoom(x)
        x = tf.image.resize(x, size=target_size)
        return x


z_dim = 128
z_vis = tf.random.normal((16, z_dim))

if __name__ == "__main__":
    generator = Generator(out_channels=3, dim=128)
    discriminator = Discriminator(dim=128)

    # # build model before loading weights
    # generated = generator(z_vis)
    # discriminator(generated)

    # generator.load_weights("saved_models/faces/generator.keras")
    # discriminator.load_weights("saved_models/faces/discriminator.keras")

    gan = WGAN(generator, discriminator, z_dim=z_dim)

    ds = get_dataset()
    ds_val = get_validation_dataset()

    generator_losses = []
    discriminator_losses = []

    progress_bar = tqdm.trange(200000)
    total_critic_steps = 0
    for iteration in progress_bar:
        # reshuffle here
        for critic_step, real_data in enumerate(ds):
            discriminator_metrics = gan.discriminator_train_step(real_data)
            for metric_name, metric_value in discriminator_metrics.items():
                tf.summary.scalar(metric_name, data=metric_value.numpy(), step=total_critic_steps)
            discriminator_losses.append(discriminator_metrics["discriminator_loss"].numpy())

            if critic_step % 5 == 4:
                generator_loss = gan.generator_train_step(batch_size)
                generator_losses.append(generator_loss.numpy())
                tf.summary.scalar("generator_loss", data=generator_loss, step=total_critic_steps)

            desc = f"generator loss: {np.mean(generator_losses[-100:]):05f} discriminator loss: {np.mean(discriminator_losses[-100:]):05f}"
            progress_bar.set_description(desc)
            if critic_step % 100 == 0:
                gan.generator.save("saved_models/faces/generator2.keras")
                gan.discriminator.save("saved_models/faces/discriminator2.keras")

                generated = gan.generator(z_vis)
                tf.summary.image("Generated", (generated + 1) / 2, max_outputs=16, step=total_critic_steps)
                tf.summary.image("Real", (real_data + 1) / 2, max_outputs=16, step=total_critic_steps)

            total_critic_steps += 1

        discriminator_metrics_all = defaultdict(list)
        for val_data in ds_val:
            discriminator_loss = gan.discriminator_val_step(val_data)
            discriminator_losses.append(discriminator_loss["discriminator_loss"].numpy())
            for metric_name, metric_value in discriminator_loss.items():
                discriminator_metrics_all[metric_name].append(metric_value.numpy())
        for metric_name, metric_values in discriminator_metrics_all.items():
            tf.summary.scalar("val_" + metric_name, data=np.mean(metric_values), step=total_critic_steps)
