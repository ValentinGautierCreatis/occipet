#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from occipet.utils import mr_forward_opp, mr_backward_opp, back_projection, forward_projection


class GradientPet:
    def __init__(self, y_pet, projector_id):
        self.y_pet = y_pet
        self.projector_id = projector_id

    def __call__(self, x):
        _, sensitivity = back_projection(np.ones(self.y_pet.shape), self.projector_id)
        _, x_proj = forward_projection(x, self.projector_id)
        ratio = self.y_pet/x_proj
        _, res = back_projection(ratio , self.projector_id)
        return res - sensitivity

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

    def generate_images_dps_brain(self, y_pet, y_mr, f_grad_pet, f_grad_mr, num_images=16, eta=0.02):
        # 1. Randomly sample noise (starting point for reverse process)
        samples = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, self.img_channels), dtype=tf.float32
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            print("sampling")
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            print("pred noise")
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            print("x0")
            x0 = self.gdf_util.predict_start_from_noise(samples, tt, pred_noise)
            print("samples")
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=False
            )
            #DPS
            print("grad pet")
            grad_pet = tf.expand_dims(f_grad_pet(x0[...,0], y_pet), axis=-1)
            grad_mr = tf.expand_dims(f_grad_mr(x0[...,1], y_mr), axis=-1)

            grad = tf.concat([grad_pet, grad_mr], axis=-1)
            samples = samples - eta*grad
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

    def generate_images_dps_brain(self, num_images=16, eta=0.02):
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
            grad_pet = tf.expand_dims(self.f_grad_pet(x0[0,...,0].numpy()), axis=-1)
            grad_mr = tf.expand_dims(self.f_grad_mr(x0[0,...,1].numpy()), axis=-1)

            grad = tf.concat([grad_pet, grad_mr], axis=-1)
            samples = samples - eta*grad
        # 3. Return generated samples
        return samples
