import tensorflow as tf


class WGAN:
    def __init__(self, generator, discriminator, z_dim=128, lam=10):
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.lam = lam

        self.generator_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=1e-4)

    @tf.function
    def generator_train_step(self, batch_size):
        z = tf.random.normal([batch_size, self.z_dim])
        with tf.GradientTape() as tape:
            fake_data = self.generator(z)
            disc_fake = self.discriminator(fake_data)
            gen_cost = -tf.reduce_mean(disc_fake)

        self.generator_optimizer.minimize(
            gen_cost, self.generator.trainable_variables, tape=tape)

        return gen_cost

    @tf.function
    def discriminator_train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        z = tf.random.normal([batch_size, self.z_dim])

        with tf.GradientTape() as tape:
            fake_data = self.generator(z)
            disc_fake = self.discriminator(fake_data)
            disc_real = self.discriminator(real_data)

            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            gradient_penalty = self._get_gradient_penalty(
                real_data, fake_data)
            disc_cost += self.lam*gradient_penalty

        self.discriminator_optimizer.minimize(
            disc_cost, self.discriminator.trainable_variables, tape=tape)

        return disc_cost

    def _get_gradient_penalty(self, real_data, fake_data):
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform(
            shape=[batch_size, 1, 1, 1],
            minval=0.,
            maxval=1.
        )
        differences = tf.stop_gradient(fake_data) - real_data
        interpolates = real_data + (alpha*differences)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(interpolates)
            disc_interpolates = self.discriminator(interpolates)

        [gradients] = tape.gradient(disc_interpolates, [interpolates])
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(slopes-1.))
        return gradient_penalty
