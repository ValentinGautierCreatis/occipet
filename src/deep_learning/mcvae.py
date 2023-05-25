#!/usr/bin/env python3
from .test_vae import Vae
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class Mcvae(tf.keras.Model):
    def __init__(self, latent_dim, beta, original_shape=(256,256,2), **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.original_dim = original_shape
        self.nb_channels = original_shape[-1]
        self.vae = [Vae(tuple(original_shape[:-1]) + (1,), latent_dim, beta) for _ in range(self.nb_channels)]
        for vae in self.vae:
            vae.build((None,) + tuple(original_shape[:-1]) + (1,))
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def encode(self, x):
        z_params = []
        z = []
        for i in range(self.nb_channels):
            z_mean, z_var, z_sampled = self.vae[i].encoder(x[..., i:i+1])
            z_params.append((z_mean, z_var))
            z.append(z_sampled)

        return z_params, z


    def decode(self, z):
        p = []
        for i in range(self.nb_channels):
            pi = [self.vae[i].decoder(z[j]) for j in range(self.nb_channels)]
            p.append(pi)
            del pi

        return p

    def call(self, x):
        _, z = self.encode(x)
        p = self.decode(z)

        return p

    def compute_kl_single(self, z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)

        return kl_loss

    def compute_kl(self, z_params):

        total = 0
        for z_mean, z_log_var in z_params:
            total += self.compute_kl_single(z_mean, z_log_var)

        return tf.reduce_mean(total)

    def compute_reconstruction_loss_single(self, reconstructed, ref):
        loss = tf.reduce_mean(tf.square(ref - reconstructed), axis=(1, 2, 3))
        return loss

    def compute_reconstruction_loss(self, p, x):
        total = 0
        for i in range(self.nb_channels):
            for j in range(self.nb_channels):
                total += self.compute_reconstruction_loss_single(p[i][j], x[..., i:i+1])
        print(tf.shape(total))
        return tf.reduce_mean(total)

    def train_step(self, data):
        x, _ = data

        with tf.GradientTape(persistent=True) as tape:
            z_params, z = self.encode(x)
            p = self.decode(z)

            kl_loss = self.beta * self.compute_kl(z_params)
            reconstruction_loss = self.compute_reconstruction_loss(p, x)

            total_loss = kl_loss + reconstruction_loss

        grads_vae0 = tape.gradient(total_loss, self.vae[0].trainable_weights)
        grads_vae1 = tape.gradient(total_loss, self.vae[1].trainable_weights)
        self.optimizer.apply_gradients(
                zip(grads_vae0, self.vae[0].trainable_weights),
        )
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

