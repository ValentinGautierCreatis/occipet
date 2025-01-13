#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .utils import ConvBlock, DeconvBlock
from occipet.utils import mr_forward_opp, mr_backward_opp, back_projection, forward_projection


class GradientPet:
    def __init__(self, y_pet, projector_id):
        self.y_pet = y_pet
        self.projector_id = projector_id

    # def __call__(self, x):
    #     _, sensitivity = back_projection(np.ones(self.y_pet.shape), self.projector_id)
    #     _, x_proj = forward_projection(x, self.projector_id)
    #     ratio = self.y_pet/(x_proj+10**(-6))
    #     _, res = back_projection(ratio , self.projector_id)
    #     return -(res - sensitivity)

    def __call__(self, x):
        _, Ax = forward_projection(x, self.projector_id)
        y_prime = Ax - self.y_pet
        product = y_prime * self.y_pet
        _, res = back_projection(product, self.projector_id)
        return res.astype(np.float32)

    # def __call__(self, x):
    #     _, Ax = forward_projection(x, self.projector_id)
    #     difference = (Ax - self.y_pet)/(np.abs(self.y_pet) + 10**(-6))
    #     _, back_diff = back_projection(difference, self.projector_id)
    #     return back_diff

class GradientMR:
    def __init__(self, y_mr, subsampling):
        self.y_mr = y_mr
        self.forward = mr_forward_opp(subsampling)
        self.backward = mr_backward_opp()

    def __call__(self, x):
        return (self.backward(self.forward(x)) - self.backward(self.y_mr)).astype(np.float32)


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999, img_size=64, img_channels=2):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, images):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t], training=True)
            print(pred_noise.shape)

            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}


    def scale_x0(self, x0, ref_mean, ref_std):
        standard_x0 = (x0 - np.mean(x0))/np.std(x0)
        return standard_x0*ref_std + ref_mean


    def generate_images(self, num_images=16):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=False
            )
        # 3. Return generated samples
        return samples

    def generate_images_dps_mnist(self, y, num_images=16, eta=0.02):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            x0 = self.gdf_util.predict_start_from_noise(samples, tt, pred_noise)
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=False
            )
            #DPS
            grad = x0 - y
            samples = samples - eta*grad
        # 3. Return generated samples
        return samples

    def generate_mnist_poisson(self,y, ref_mean, ref_std, num_images=1, eta1=0.02, eta2=0.02):
        # First channel has poisson noise and the second Gaussian noise
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            x0 = self.gdf_util.predict_start_from_noise(samples, tt, pred_noise)
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=False
            )

            scaled_x0 = self.scale_x0(x0, ref_mean, ref_std)

            #DPS
            grad_poisson = eta1 * tf.expand_dims(scaled_x0[...,0].numpy()/(y[...,0] + 10**(-4)) - 1, axis=-1)
            grad_gaussian = eta2 * tf.expand_dims(scaled_x0[...,1].numpy() - y[...,1], axis=-1)
            grad = tf.concat([grad_poisson, grad_gaussian], axis=-1)

            samples = samples - grad
        # 3. Return generated samples
        return samples

    def plot_images(
        self, epoch=None, logs=None, num_rows=2, num_cols=8, figsize=(12, 5)
    ):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_images(num_images=num_rows * num_cols)
        generated_samples = (
            # tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
            generated_samples
            .numpy()
            .astype(np.uint8)
        )

        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.show()


class DiffusionModelBrain(DiffusionModel):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999, img_size=64, img_channels=2):
        super().__init__(network, ema_network, timesteps, gdf_util, ema, img_size, img_channels)

    def scale_x0(self, x0, ref_mean, ref_std):
        standard_x0 = (x0 - np.mean(x0))/np.std(x0)
        return standard_x0*ref_std + ref_mean

    def generate_images_dps_brain(self, f_grad_pet, f_grad_mr, ref_mean, ref_std, num_images=1, eta1=0.02, eta2=0.02):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            x0 = self.gdf_util.predict_start_from_noise(samples, tt, pred_noise)
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=False
            )
            #DPS
            scaled_x0 = self.scale_x0(x0, ref_mean, ref_std)
            grad_pet = eta1 * tf.expand_dims(f_grad_pet(scaled_x0[0,...,0].numpy()), axis=-1)
            grad_mr = eta2 * tf.expand_dims(f_grad_mr(scaled_x0[0,...,1].numpy()), axis=-1)

            grad = tf.concat([grad_pet, grad_mr], axis=-1)
            samples = samples - grad
        # 3. Return generated samples
        return samples


class DiffusionModelMono(DiffusionModel):
    def __init__(self, network, f_grad, ema_network, timesteps, gdf_util, ema=0.999, img_size=64, img_channels=2):
        self.f_grad = f_grad
        super().__init__(network, ema_network, timesteps, gdf_util, ema, img_size, img_channels)

    # Only num_images=1 or breaks
    def generate_images_dps_brain(self, num_images=1, eta=0.02):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            x0 = self.gdf_util.predict_start_from_noise(samples, tt, pred_noise)
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=False
            )
            #DPS
            grad = tf.expand_dims(self.f_grad(x0[0,...,0].numpy()), axis=-1)

            samples = samples - eta*grad
        # 3. Return generated samples
        return samples



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        dim = tf.shape(z_mean)
        epsilon = tf.keras.backend.random_normal(shape=dim)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=2, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv = keras.Sequential(
            [
                ConvBlock(32),
                ConvBlock(64)
            ]
        )
        self.mean = ConvBlock(latent_dim)
        self.log_var = ConvBlock(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv(inputs)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, latent_dim = 2, nb_channels=2, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.deconv = keras.Sequential(
            [
                DeconvBlock(latent_dim),
                DeconvBlock(64),
                DeconvBlock(32),
                layers.Conv2DTranspose(
                    filters=nb_channels, kernel_size=3, strides=1, padding="same"
                ),
            ]
        )

    def call(self, inputs):

        return self.deconv(inputs)


class Vae(tf.keras.Model):
    def __init__(self, img_size=(256,256,2), latent_dim=2, beta=1.0, sparse=False, name="vae", **kwargs):
        super(Vae, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.encoder = Encoder(latent_dim=latent_dim)
        self.img_size = img_size
        self.encoder.compute_output_shape((None,) + tuple(img_size))
        self.decoder = Decoder(latent_dim, img_size[-1])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @tf.function
    def encode(self, x):
        return self.encoder(x)

    @tf.function
    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        # self._set_inputs(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z_mean)
        return reconstructed

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # data = tf.convert_to_tensor(data)
            x = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_mean(tf.square(x - reconstruction), axis=(1, 2, 3))
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1,2,3)))
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
