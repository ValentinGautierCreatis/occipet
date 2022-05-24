#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

tf.keras.backend.clear_session()  # For easy reset of notebook state.


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
                  keras.losses.mean_squared_error(x, reconstruction), axis=(1, 2)
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


# original_dim = 784
# vae = VariationalAutoEncoder(original_dim, 64, 32)  #, input_shape=(784,)

# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# mse_loss_fn = tf.keras.losses.MeanSquaredError()

# loss_metric = tf.keras.metrics.Mean()

# (x_train, _), _ = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype('float32') / 255

# train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# # Iterate over epochs.
# for epoch in range(1):
#   print('Start of epoch %d' % (epoch,))

#   # Iterate over the batches of the dataset.
#   for step, x_batch_train in enumerate(train_dataset):
#     with tf.GradientTape() as tape:
#       # !!! uncomment the following two lines to use workaround and skip !!!
#       # if step == 0 and epoch == 0:
#       #   vae._set_inputs(x_batch_train)
#       reconstructed = vae(x_batch_train)
#       # Compute reconstruction loss
#       loss = mse_loss_fn(x_batch_train, reconstructed)
#       loss += sum(vae.losses)  # Add KLD regularization loss

#     grads = tape.gradient(loss, vae.trainable_weights)
#     optimizer.apply_gradients(zip(grads, vae.trainable_weights))

#     loss_metric(loss)

#     if step % 100 == 0:
#       print('step %s: mean loss = %s' % (step, loss_metric.result()))

# vae.save('vae')
