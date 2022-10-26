#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D

from .utils import *

tf.keras.backend.clear_session()  # For easy reset of notebook state.


def l2_loss(x, y):
  """Computes the l2 loss between 2 batches of images of
  shape (None, w, h, c)



  Parameters
  ----------
  x : image 1
  y : image 2

  """

  reconstruction_loss = tf.reduce_mean(
    tf.reduce_sum(
      tf.square(x - y), axis=(1,2,3)
    )
  )

  return reconstruction_loss


class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    dim = tf.shape(z_mean)
    epsilon = tf.keras.backend.random_normal(shape=dim)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
               latent_dim=32,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.conv = keras.Sequential([
        layers.Conv2D(
              filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
        layers.Conv2D(
              filters=64, kernel_size=3, strides=(2, 2), activation='relu')
      ]
    )
    self.flatten = layers.Flatten()
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.conv(inputs)
    x = self.flatten(x)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self,
               original_dim,
               name='decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense = layers.Dense(original_dim[0]//4 * original_dim[1]//4 * 32)
    self.reshape =  layers.Reshape(target_shape=(original_dim[0]//4, original_dim[1]//4, 32))
    self.deconv = keras.Sequential([
      layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=2, padding='same',
        activation='relu'),
      layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=2, padding='same',
        activation='relu'),
      layers.Conv2DTranspose(filters=original_dim[2], kernel_size=3, strides=1, padding='same')
    ])

  def call(self, inputs):
    x = self.dense(inputs)
    x = self.reshape(x)

    return self.deconv(x)


class VariationalAutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               original_dim,
               latent_dim=32,
               name='autoencoder',
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.original_dim = original_dim
    self.encoder = Encoder(latent_dim=latent_dim)
    self.decoder = Decoder(original_dim)
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
            tf.reduce_sum(
              tf.square(x - reconstruction), axis=(1,2,3)
            )
          )

          kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
          kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
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
    return self.train_step(data)


class BetaVAE(VariationalAutoEncoder):
  def __init__(self, original_dim, latent_dim=32, beta=1, name='autoencoder',
               **kwargs):
    super().__init__(original_dim, latent_dim, name, **kwargs)
    self.beta = beta

  def train_step(self, data):
      with tf.GradientTape() as tape:
          # data = tf.convert_to_tensor(data)
          x, _ = data
          z_mean, z_log_var, z = self.encoder(x)
          reconstruction = self.decoder(z)
          reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
              tf.square(x - reconstruction), axis=(1,2,3)
            )
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


def bimodal_loss(mod1, mod2, label1, label2):

    loss1 = tf.reduce_mean(
      tf.reduce_sum(tf.square(mod1 - label1), axis=(1,2,3))
    )
    loss2 = tf.reduce_mean(
      tf.reduce_sum(tf.square(mod2 - label2), axis=(1,2,3))
    )

    return loss1, loss2


class BimodalVAE(BetaVAE):
  def __init__(self, original_dim, latent_dim=32, beta=1, mod1_weight=1,
               name='autoencoder', **kwargs):
    super().__init__(original_dim, latent_dim, beta, name, **kwargs)
    self.mod1_weight = mod1_weight
    self.mod1_loss_tracker = keras.metrics.Mean(name="mod1 loss")
    self.mod2_loss_tracker = keras.metrics.Mean(name="mod2 loss")

  def train_step(self, data):
      with tf.GradientTape() as tape:
          # data = tf.convert_to_tensor(data)
          x, _ = data
          z_mean, z_log_var, z = self.encoder(x)
          reconstruction = self.decoder(z)

          mod1, mod2 = tf.split(reconstruction, 2, axis=-1)
          label1, label2 = tf.split(x, 2, axis=-1)
          loss1, loss2 = bimodal_loss(mod1, mod2, label1, label2)

          reconstruction_loss = self.mod1_weight * loss1 + loss2

          kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
          kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
          total_loss = reconstruction_loss + self.beta*kl_loss

      grads = tape.gradient(total_loss, self.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.kl_loss_tracker.update_state(kl_loss)
      self.mod1_loss_tracker.update_state(self.mod1_weight * loss1)
      self.mod2_loss_tracker.update_state(loss2)
      return {
          "total_loss": self.total_loss_tracker.result(),
          "reconstruction_loss": self.reconstruction_loss_tracker.result(),
          "loss 1": self.mod1_loss_tracker.result(),
          "loss 2": self.mod2_loss_tracker.result(),
          "kl_loss": self.kl_loss_tracker.result(),
      }


class Encoder2(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self, block_config, channel_config,
               name='encoder', **kwargs):
    super(Encoder2, self).__init__(name=name, **kwargs)
    self.first_conv = Conv2D(filters=64, kernel_size=3, strides=1,
                             padding='same')
    blocks = []
    for (resolution, down_rate) in block_config:
      if isinstance(resolution, tuple):
        # Denotes transition to another resolution
        res1, res2 = resolution
        blocks.append(
          Conv2D(channel_config[res2], 1)
        )

      else:
        nb_filters = channel_config[resolution]
        use_3x3 = resolution > 1
        blocks.append(
          ResBlock(
            int(0.5 * nb_filters),
            nb_filters,
            down_rate=down_rate,
            residual=True,
            use_3x3=use_3x3
          )
        )

      self.blocks = keras.Sequential(blocks)
      self.mu = Conv2D(channel_config[1], 1)
      self.logvar = Conv2D(channel_config[1], 1)
      self.sampling = Sampling()


  def call(self, inputs):
    x = self.first_conv(inputs)
    x = self.blocks(x)
    z_mean = self.mu(x)
    z_logvar = self.logvar(x)
    z = self.sampling((z_mean, z_logvar))
    return z_mean, z_logvar, z


class Decoder2(layers.Layer):
  def __init__(self, original_depths, block_config, channel_config,
               name='encoder', **kwargs):
    super(Decoder2, self).__init__(name=name, **kwargs)
    blocks = []
    for (resolution, up_rate) in block_config:
      if isinstance(resolution, tuple):
        # Denotes transition to another resolution
        res1, res2 = resolution
        blocks.append(
          Conv2D(channel_config[res2], 1)
        )
      elif up_rate is not None:
        blocks.append(UpSampling2D(up_rate))
      else:
        nb_filters = channel_config[resolution]
        use_3x3 = resolution > 1
        blocks.append(
          ResBlock(
            int(0.5 * nb_filters),
            nb_filters,
            down_rate=None,
            residual=True,
            use_3x3=use_3x3
          )
        )

    self.blocks = keras.Sequential(blocks)
    self.last_conv = Conv2D(original_depths, 3, 1, "same", activation='sigmoid')


  def call(self, inputs):
    x = self.blocks(inputs)
    x = self.last_conv(x)
    return x


class Vae2(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               original_depth,
               encoder_block,
               decoder_block,
               encoder_channel,
               decoder_channel,
               beta,
               name='autoencoder',
               **kwargs):
    super(Vae2, self).__init__(name=name, **kwargs)
    self.original_depth = original_depth
    self.encoder = Encoder2(encoder_block, encoder_channel)
    self.decoder = Decoder2(original_depth, decoder_block, decoder_channel)
    self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = keras.metrics.Mean(
        name="reconstruction_loss"
    )
    self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    self.beta = beta



  def call(self, inputs):
    _, _, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    return reconstructed


  def train_step(self, data):
      with tf.GradientTape() as tape:
          # data = tf.convert_to_tensor(data)
          x, _ = data
          z_mean, z_log_var, z = self.encoder(x)
          reconstruction = self.decoder(z)
          reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
              tf.square(x - reconstruction), axis=(1,2,3)
            )
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
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 # save_weights_only=True,
                                                 # verbose=1)

# Train the model with the new callback
# model.fit(train_images,
#           train_labels,
#           epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.
