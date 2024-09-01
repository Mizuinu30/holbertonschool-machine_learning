#!/usr/bin/env python3
""" This module contains the WGAN class that inherits from keras.Model """


def train_step(self, useless_argument):
    """ Overloading train_step()"""
    # Train the discriminator
    for _ in range(self.disc_iter):
        with tf.GradientTape() as tape:
            real_samples = self.get_real_sample()
            fake_samples = self.get_fake_sample(training=True)
            interpolated_samples = self.get_interpolated_sample(real_samples, fake_samples)

            real_output = self.discriminator(real_samples, training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            discr_loss = self.discriminator.loss(real_output, fake_output, self.gradient_penalty(interpolated_samples))

        grads = tape.gradient(discr_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

    # Train the generator
    with tf.GradientTape() as tape:
        fake_samples = self.get_fake_sample(training=True)
        fake_output = self.discriminator(fake_samples, training=True)
        gen_loss = self.generator.loss(fake_output)

    grads = tape.gradient(gen_loss, self.generator.trainable_variables)
    self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

    # Compute gradient penalty for the last batch of samples
    gp = self.gradient_penalty(self.get_interpolated_sample(real_samples, fake_samples))

    return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
