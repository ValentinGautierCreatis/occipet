#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    ReLU,
    BatchNormalization,
    MaxPool2D,
)

class ConvBlock(keras.layers.Layer):
    # Stride is 2 so divide size by 2 each time
    def __init__(self, nb_filters, **kwargs) -> None:
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = Conv2D(nb_filters, kernel_size=(3, 3), padding="same", strides=2)
        self.normalize = BatchNormalization()
        self.activation = ReLU()

    def call(self, x):
        xhat = self.conv(x)
        xhat = self.normalize(xhat)
        xhat = self.activation(xhat)
        return xhat


class DeconvBlock(keras.layers.Layer):
    # Stride is 2 so divide size by 2 each time
    def __init__(self, nb_filters, **kwargs) -> None:
        super(DeconvBlock, self).__init__(**kwargs)
        self.deconv = Conv2DTranspose(
            nb_filters, kernel_size=(3, 3), padding="same", strides=2
        )
        self.normalize = BatchNormalization()
        self.activation = ReLU()

    def call(self, x):
        xhat = self.deconv(x)
        xhat = self.normalize(xhat)
        xhat = self.activation(xhat)
        return xhat


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
        # self.dense = layers.Dense(128)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv(inputs)
        self.shape_before_flatten = keras.backend.int_shape(x)[1:]
        x = self.flatten(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, shape_before_flatten, nb_channels=2, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense = layers.Dense(np.prod(shape_before_flatten))
        self.reshape = layers.Reshape(target_shape=(shape_before_flatten))
        self.deconv = keras.Sequential(
            [
                DeconvBlock(64),
                DeconvBlock(32),
                layers.Conv2DTranspose(
                    filters=nb_channels, kernel_size=3, strides=1, padding="same"
                ),
            ]
        )

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)

        return self.deconv(x)


class Vae(tf.keras.Model):
    def __init__(
        self,
        original_shape=(256, 256, 2),
        latent_dim=32,
        beta=1,
        alpha=0.9,
        name="vae",
        **kwargs
    ):
        super(Vae, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.alpha = alpha
        self.pet_weight = tf.Variable(initial_value=20000.0, trainable=False, dtype=tf.float64)
        self.mr_weight = tf.Variable(initial_value=10000.0, trainable=False, dtype=tf.float64)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.original_shape = original_shape
        self.encoder.compute_output_shape((None,) + tuple(original_shape))
        self.decoders = [
            Decoder(self.encoder.shape_before_flatten, 1),
            Decoder(self.encoder.shape_before_flatten, 1),
        ]
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.pet_loss_tracker = keras.metrics.Mean(name="pet_loss")
        self.mr_loss_tracker = keras.metrics.Mean(name="mr_loss")

    @tf.function
    def encode(self, x):
        return self.encoder(x)

    @tf.function
    def decode(self, z):
        reconstruction = tf.concat(
            [self.decoders[0](z), self.decoders[1](z)], axis=-1
        )
        return reconstruction

    def call(self, inputs):
        z_mean, *_ = self.encoder(inputs)
        reconstructed = tf.concat(
            [self.decoders[0](z_mean), self.decoders[1](z_mean)], axis=-1
        )
        return reconstructed

    @tf.function
    def compute_weight_update(self, grads_ref, grads_modality, weight):
        grads_ref = [grad for grad in grads_ref if grad is not None]
        flat_gradients_ref = [tf.reshape(grad, (tf.math.reduce_prod(tf.shape(grad)), )) for grad in grads_ref]
        all_gradients_ref = tf.concat(flat_gradients_ref, axis=0)
        max_gradient_ref = tf.reduce_max(tf.abs(all_gradients_ref))

        grads_modality = [grad for grad in grads_modality if grad is not None]
        flat_gradients = [tf.reshape(grad, (tf.math.reduce_prod(tf.shape(grad)), )) for grad in grads_modality]
        all_gradients = tf.concat(flat_gradients, axis=0)
        mean_gradient_mod = tf.reduce_mean(tf.abs(all_gradients))

        temp = max_gradient_ref/mean_gradient_mod

        return (1 - self.alpha) * weight + self.alpha * temp

    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            x, _ = data
            z_mean, z_log_var, z = self.encoder(x)

            pet_decoded = self.decoders[0](z)
            pet_loss = self.pet_weight * tf.reduce_mean(
                tf.reduce_mean(tf.square(x[...,0:1] - pet_decoded), axis=(1, 2, 3))
            )

            mr_decoded = self.decoders[1](z)
            mr_loss = self.mr_weight * tf.reduce_mean(
                tf.reduce_mean(tf.square(x[...,1:] - mr_decoded), axis=(1, 2, 3))
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            reconstruction_loss = (pet_loss + mr_loss)


            total_loss = reconstruction_loss + kl_loss

        kl_grads = tape.gradient(kl_loss, self.trainable_weights)
        pet_grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        mr_grads = tape.gradient(mr_loss, self.trainable_weights)

        self.pet_weight.assign(self.compute_weight_update(kl_grads, pet_grads, self.pet_weight))
        self.mr_weight.assign(self.compute_weight_update(kl_grads, mr_grads, self.mr_weight))

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.pet_loss_tracker.update_state(pet_loss)
        self.mr_loss_tracker.update_state(mr_loss)

        del tape

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "pet_loss": self.pet_loss_tracker.result(),
            "mr_loss": self.mr_loss_tracker.result(),
            "pet_weight": self.pet_weight,
            "mr_weight": self.mr_weight,
        }

    def test_step(self, data):
        # Decoded on z_mean
        x, _ = data
        z_mean, z_log_var, _ = self.encoder(x)
        reconstruction = tf.concat(
            [self.decoders[0](z_mean), self.decoders[1](z_mean)], axis=-1
        )

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(x - reconstruction), axis=(1, 2, 3))
        )

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
