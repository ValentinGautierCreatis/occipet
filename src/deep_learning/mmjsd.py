
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
        name="vae",
        **kwargs
    ):
        super(Vae, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.encoder = Encoder(latent_dim=latent_dim)
        self.original_shape = original_shape
        self.encoder.compute_output_shape((None,) + tuple(original_shape))
        self.decoder = Decoder(self.encoder.shape_before_flatten, 1)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class mmJSD(tf.keras.Model):
    def __init__(self,
        original_shape=(256, 256, 2),
        latent_dim=32,
        beta=1,
        name="vae",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.nb_modalities = original_shape[-1]
        self.vaes = [Vae(tuple(original_shape[:-1])+(1,), latent_dim, beta, name) for _ in range(self.nb_modalities)]
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.elbo_poe_tracker = keras.metrics.Mean(name="elbo_poe")
        self.jensen_tracker = keras.metrics.Mean(name="jensen")
        self.reconstruction_poe_tracker = keras.metrics.Mean(name="reconstruction_poe")

    def poe(self, means, log_vars):
        Ts = [tf.math.exp(-lv) for lv in log_vars]
        log_var = -tf.math.log(sum(Ts))

        mean_numerator = sum([Ti * mean_i for (Ti,mean_i) in zip(Ts, means)])
        mean = mean_numerator * tf.math.exp(log_var)

        return mean, log_var

    def encode(self, x):
        mean1, log_var1, z1 = self.vaes[0].encode(x[..., :1])
        mean2, log_var2, z2 = self.vaes[1].encode(x[..., :-1])

        # mean, log_var = self.poe(mean1, mean2, log_var1, log_var2)
        # mean, log_var = self.poe(mean, tf.zeros_like(mean), log_var, tf.zeros_like(log_var))
        mean, log_var = self.poe([mean1, mean2], [log_var1, log_var2])
        z = Sampling()((mean, log_var))

        return mean1, log_var1, z1, mean2, log_var2, z2, mean, log_var, z

    def decode(self, z):
        reconstruction = tf.concat(
            [self.vaes[0].decode(z), self.vaes[1].decode(z)], axis=-1
        )
        return reconstruction

    def call(self, x):
        _,_,_,_,_,_,mean,_,_ = self.encode(x)
        return self.decode(mean)

    @staticmethod
    def kl_loss(z_mean1, z_log_var1, z_mean2, z_log_var2):
        kl = -0.5 * (1 - z_log_var2 + z_log_var1 - tf.square(z_mean1 - z_mean2)/tf.exp(z_log_var2) - tf.exp(z_log_var1 - z_log_var2))
        kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        return kl

    @staticmethod
    def reconstruction_loss(x_pred, x_ref):
        loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(x_pred - x_ref), axis=(1, 2, 3))
        )
        return loss


    def jensen_loss(self, mean1, log_var1, mean2, log_var2, mean_prior, log_var_prior):

        mean_dynamic_prior, log_var_dynamic_prior = self.poe([mean1, mean2, mean_prior], [log_var1, log_var2, log_var_prior])

        kl1 = self.kl_loss(mean1, log_var1, mean_dynamic_prior, log_var_dynamic_prior)
        kl2 = self.kl_loss(mean2, log_var2, mean_dynamic_prior, log_var_dynamic_prior)
        kl_prior = self.kl_loss(mean_prior, log_var_prior, mean_dynamic_prior, log_var_dynamic_prior)
        return kl1 + kl2 + kl_prior


    def train_step(self, data):
        x, _ = data
        x1 = x[...,:1]
        x2 = x[...,:-1]
        with tf.GradientTape(persistent=True) as tape:
            z_mean1, z_log_var1, z1, z_mean2, z_log_var2, z2, z_mean, z_log_var, z = self.encode(x)
            mean_prior = tf.zeros_like(z_mean)
            log_var_prior = tf.zeros_like(z_log_var)

            decoded_poe = self.decode(z)
            # decoded1 = self.decode(z1)
            # decoded2 = self.decode(z2)

            # reconstruction1 = self.reconstruction_loss(decoded1, x1)
            # reconstruction2 = self.reconstruction_loss(decoded2, x2)
            reconstruction_poe = self.reconstruction_loss(decoded_poe, x)

            jensen = self.jensen_loss(z_mean1, z_log_var1, z_mean2, z_log_var2, mean_prior, log_var_prior)

            loss_poe = reconstruction_poe + self.beta * jensen

            total_loss = loss_poe# + loss1 + loss2

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.elbo_poe_tracker.update_state(loss_poe)
        self.jensen_tracker.update_state(jensen)
        self.reconstruction_poe_tracker.update_state(reconstruction_poe)

        return {
            "elbo_poe": self.elbo_poe_tracker.result(),
            "reconstruction_poe": self.reconstruction_poe_tracker.result(),
            "jensen": self.jensen_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
        }
