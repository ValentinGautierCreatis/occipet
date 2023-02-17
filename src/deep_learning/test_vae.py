#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.layers.convolutional.conv2d import Conv2D
from keras.layers.convolutional.conv2d_transpose import Conv2DTranspose
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from .utils import ConvBlock, DeconvBlock


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        dim = tf.shape(z_mean)
        epsilon = tf.keras.backend.random_normal(shape=dim)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv = keras.Sequential([ConvBlock(32), ConvBlock(64)])
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(64)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv(inputs)
        self.shape_before_flatten = keras.backend.int_shape(x)[1:]
        x = self.flatten(x)
        x = self.dense(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, shape_before_flatten, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense = layers.Dense(np.prod(shape_before_flatten))
        self.reshape = layers.Reshape(target_shape=(shape_before_flatten))
        self.deconv = keras.Sequential([DeconvBlock(64), DeconvBlock(32)])
        self.last_conv = Conv2DTranspose(
            2, kernel_size=(3, 3), padding="same", strides=1
        )

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.deconv(x)
        return self.last_conv(x)


class Vae(tf.keras.Model):
    def __init__(
        self, latent_dim=32, beta=1.0, sparse=False, name="vae", **kwargs
    ):
        super(Vae, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.encoder = Encoder(latent_dim=latent_dim)
        self.encoder.compute_output_shape((None ,256,256,2))
        self.decoder = Decoder(self.encoder.shape_before_flatten)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        # self._set_inputs(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        # kl_loss = - 0.5 * tf.reduce_sum(
        #     z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        # self.add_loss(kl_loss)
        return reconstructed

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # data = tf.convert_to_tensor(data)
            x, _ = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_mean(tf.square(x - reconstruction), axis=(1, 2, 3))
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, _ = data
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(x - reconstruction), axis=(1, 2, 3))
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta * tf.reduce_sum(tf.reduce_mean(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
