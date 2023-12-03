import tensorflow as tf


class WGAN:
    def __init__(self, generator, discriminator, z_dim=128, lam=10, lr=1e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.lam = lam

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0, beta_2=0.9)

    @tf.function
    def generator_train_step(self, batch_size):
        z = tf.random.normal([batch_size, self.z_dim])
        with tf.GradientTape() as tape:
            fake_data = self.generator(z, training=True)
            disc_fake = self.discriminator(fake_data, training=True)
            gen_cost = -tf.reduce_mean(disc_fake)

        self.generator_optimizer.minimize(
            gen_cost, self.generator.trainable_variables, tape=tape)

        return gen_cost

    @tf.function
    def discriminator_train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        z = tf.random.normal([batch_size, self.z_dim])

        with tf.GradientTape() as tape:
            fake_data = self.generator(z, training=True)
            disc_fake = self.discriminator(fake_data, training=True)
            disc_real = self.discriminator(real_data, training=True)

            em_score = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            gradient_penalty = self._get_gradient_penalty(
                real_data, fake_data)
            disc_cost = em_score + self.lam*gradient_penalty

        self.discriminator_optimizer.minimize(
            disc_cost, self.discriminator.trainable_variables, tape=tape)

        return {
            "em_score": -em_score,
            "gradient_penalty": gradient_penalty,
            "discriminator_loss": disc_cost
        }
    
    @tf.function
    def discriminator_val_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        z = tf.random.normal([batch_size, self.z_dim])

        fake_data = self.generator(z, training=True)
        disc_fake = self.discriminator(fake_data, training=True)
        disc_real = self.discriminator(real_data, training=True)

        em_score = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        gradient_penalty = self._get_gradient_penalty(
            real_data, fake_data)
        disc_cost = em_score + self.lam*gradient_penalty

        return {
            "em_score": -em_score,
            "gradient_penalty": gradient_penalty,
            "discriminator_loss": disc_cost
        }

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
            disc_interpolates = self.discriminator(interpolates, training=True)

        [gradients] = tape.gradient(disc_interpolates, [interpolates])
        gradients=tf.reshape(gradients, (batch_size, -1))
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(slopes-1.))
        return gradient_penalty
