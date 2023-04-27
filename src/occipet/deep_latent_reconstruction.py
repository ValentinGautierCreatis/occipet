#!/usr/bin/env python3

import numpy as np
from functools import partial

import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from occipet.reconstruction import em_step
from skimage.metrics import structural_similarity as ssim
from typing import Callable
from .utils import *
import tensorflow as tf
import scipy.sparse.linalg as linalg
from deep_learning import variational_auto_encoder


class DeepLatentReconstruction:
    def __init__(
        self, autoencoder: variational_auto_encoder.VariationalAutoEncoder
    ) -> None:
        self.autoencoder = autoencoder
        self.xi = 1
        self.u = 10
        self.tau = 1
        self.tau_pet = 1
        self.tau_mr = 1
        self.tau_max = 100
        self.coeffs = np.array([1, 1])
        self.correction_mean = 0
        self.correction_std = 1
        self.rho_pet: float
        self.rho_mr: float
        self.y_pet: np.ndarray
        self.y_mr: np.ndarray
        self.N: int
        self.projector_id: int
        self.mri_subsampling: Callable
        self.nb_steps: int
        self.eps_rel: float

    def pet_step(self, current, pet_decoded, mu):
        _, sensitivity = back_projection(np.ones(self.y_pet.shape), self.projector_id)
        x_em = em_step(self.y_pet, current, self.projector_id, sensitivity)
        square_root_term = (pet_decoded - mu - (sensitivity / self.rho_pet)) ** 2 + (
            4 * x_em * sensitivity
        ) / self.rho_pet
        new_xpet = 0.5 * (
            pet_decoded - mu - (sensitivity / self.rho_pet) + np.sqrt(square_root_term)
        )

        return new_xpet

    def apply_A(self, x):
        transformed = self.N * ifft2(self.mri_subsampling(fft2(x)))
        return self.rho_mr * x + np.real(transformed)

    def apply_A_on_flattened(self, shape, x):
        x = x.reshape(shape)
        Ax = self.apply_A(x)
        return Ax.ravel()

    def mr_step(self, mr_decoded, mu):
        signal = (1 / (self.rho_mr + self.N)) * (
            self.N * ifft2(self.y_mr) + self.rho_mr * (mr_decoded - mu)
        )
        return np.real(signal)

    def mr_step_subsampled(self, x_mr, mr_decoded, mu):
        partial_A = partial(self.apply_A_on_flattened, x_mr.shape)
        A = linalg.LinearOperator((x_mr.size, x_mr.size), matvec=partial_A)
        b = np.real(ifft2(self.y_mr)) + self.rho_mr * (mr_decoded - mu)

        b_flat = b.ravel()
        x_flat = x_mr.ravel()

        x_flat, _ = linalg.cg(A, b_flat, x0=x_flat)
        return x_flat.reshape(x_mr.shape)

    def z_step(self, z, x, mu):
        def f():
            decoded = (
                self.autoencoder.decoder(z) * self.correction_std + self.correction_mean
            )
            new_image = (x + mu) - decoded
            squared = tf.math.square(new_image)
            return squared

        self.optimizer.minimize(f, [z])
        return z

        # with tf.GradientTape() as tape:
        #     tape.watch(z)
        #     decoded = (
        #         self.autoencoder.decoder(z) * self.correction_std + self.correction_mean
        #     )
        #     new_image = (x + mu) - decoded
        #     squared = tf.math.square(new_image)
        # update_term = tape.gradient(squared, z)
        # return self.optimizer.apply_gradients(zip(update_term, z))

        # return z - self.step_size * update_term

    def lagrangian_step(self, x, mu, decoded):
        return mu + x - decoded

    def compute_residuals(self, x, mu, decoded, old_decoded):
        normalisation_primal = np.max([np.linalg.norm(x), np.linalg.norm(decoded)])
        norm_primal = np.linalg.norm((x - decoded) / normalisation_primal)

        normalisation_dual = np.linalg.norm(mu)
        norm_dual = np.linalg.norm((decoded - old_decoded) / normalisation_dual)

        return norm_primal, norm_dual

    def tau_update(self, norm_primal, norm_dual):
        update_coeff = np.sqrt(norm_primal / (norm_dual * self.xi))
        if 1 <= update_coeff < self.tau_max:
            return update_coeff
        elif (1 / self.tau_max) < update_coeff < 1:
            return 1 / update_coeff
        else:
            return self.tau_max

    def compute_rho_update_factor(self, norm_primal, norm_dual):
        update_factor = 1
        if norm_primal > self.xi * self.u * norm_dual:
            # rho = min(tau * rho, rho_max)
            update_factor = self.tau
        elif norm_dual > (self.u / self.xi) * norm_primal:
            # rho = min(rho/tau, rho_max)
            update_factor = 1 / self.tau

        return update_factor

    def compute_splitted_rho_update_factor(self, norm_primal, norm_dual, tau):
        update_factor = 1
        if norm_primal > self.xi * self.u * norm_dual:
            update_factor = tau
        elif norm_dual > (self.u / self.xi) * norm_primal:
            update_factor = 1 / tau

        return update_factor

    def compute_coeffs(self, x, decoded):
        return x.mean(axis=(0, 1)) / decoded.mean(axis=(0, 1))

    def compute_correction(self, ref, decoded):
        mean_ref = np.mean(ref, axis=(0, 1), keepdims=True)
        std_ref = np.sqrt(((ref - mean_ref) ** 2).mean(axis=(0, 1), keepdims=True))

        mean_dec = np.mean(decoded, axis=(0, 1), keepdims=True)
        std_dec = np.sqrt(((decoded - mean_dec) ** 2).mean(axis=(0, 1), keepdims=True))

        return mean_ref - mean_dec, std_ref / std_dec

    def decoding(self, z):
        return np.squeeze(self.autoencoder.decoder(z).numpy(), axis=0)

    def pet_reconstruction_step(self, x, mu, z):
        decoded = self.decoding(z)
        # self.coeffs = self.compute_coeffs(x, decoded)
        self.correction_mean, self.correction_std = self.compute_correction(x, decoded)
        # decoded = self.coeffs * decoded
        decoded = self.correction_std * decoded + self.correction_mean
        # decoded[decoded[:,:,0] <= 0] = 10**-6
        # print(f"ref {x[:,:,0].mean()}")
        # print(self.rho_pet)
        pet_decoded = decoded[:, :, 0]
        x_pet = x[:, :, 0]

        # PET STEP
        new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:, :, 0])
        new_x = np.concatenate(
            (np.expand_dims(new_x_pet, axis=-1), x[:, :, 1:]), axis=-1
        )

        # Z STEP
        new_z = self.z_step(z, new_x, mu)
        new_decoded = self.decoding(new_z)
        # new_coeff = self.compute_coeffs(x, new_decoded)
        new_mean, new_std = self.compute_correction(new_x, new_decoded)
        # new_decoded = new_coeff * new_decoded
        new_decoded = new_mean + new_decoded * new_std

        # LAGRANGIAN STEP
        new_mu = self.lagrangian_step(new_x, mu, new_decoded)

        # RESIDUALS
        norm_primal, norm_dual = self.compute_residuals(
            new_x[:, :, 0], new_mu[:, :, 0], new_decoded[:, :, 0], decoded[:, :, 0]
        )

        # TAU UPDATE
        self.tau = self.tau_update(norm_primal, norm_dual)

        # RHO UPDATE
        update_factor = self.compute_rho_update_factor(norm_primal, norm_dual)
        self.rho_pet = self.rho_pet * update_factor
        # Remember to rescale Mu too
        new_mu = new_mu / update_factor

        stop = False
        if norm_primal <= self.eps_rel and norm_dual <= self.eps_rel:
            stop = True

        return new_x, new_mu, new_z, stop

    # Input images with 2 shapes
    def pet_reconstruction(
        self, x_pet0, ref_mr, y_pet, projector_id, step_size, nb_steps, eps_rel
    ):
        # Initializations
        self.optimizer = tf.keras.optimizers.Adam(0.05)
        self.nb_steps = nb_steps
        self.step_size = step_size
        self.y_pet = y_pet
        self.projector_id = projector_id
        self.eps_rel = eps_rel
        x_pet0_normalized = (x_pet0 - x_pet0.mean()) / x_pet0.std()
        ref_mr_normalized = (ref_mr - ref_mr.mean()) / ref_mr.std()
        x0_normalized = np.concatenate(
            (
                np.expand_dims(x_pet0_normalized, axis=-1),
                np.expand_dims(ref_mr_normalized, axis=-1),
            ),
            axis=-1,
        )
        *_, z = self.autoencoder.encoder(np.expand_dims(x0_normalized, axis=0))
        z = tf.Variable(z, trainable=True)

        x = np.concatenate(
            (np.expand_dims(x_pet0, axis=-1), np.expand_dims(ref_mr, axis=-1)), axis=-1
        )
        mu = np.zeros_like(x)
        self.rho_pet = 1 / np.sum(y_pet)

        # Begining of the algo
        for _ in range(self.nb_steps):
            x, mu, z, stop = self.pet_reconstruction_step(x, mu, z)
            if stop:
                break

        return x  # [:,:,0]

    def reconstruction_step(self, x, mu, z):
        decoded = self.decoding(z)
        self.correction_mean, self.correction_std = self.compute_correction(x, decoded)
        decoded = self.correction_std * decoded + self.correction_mean
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        ind = 0
        axes[0].imshow(decoded[:, :, ind])
        axes[1].imshow(x[:, :, ind])
        plt.show()

        # PET STEP
        new_x_pet = self.pet_step(x[:, :, 0], decoded[:, :, 0], mu[:, :, 0])

        # MR STEP
        # new_x_mr = self.mr_step(decoded[:, :, 1], mu[:, :, 1])
        new_x_mr = self.mr_step_subsampled(x[:, :, 1], decoded[:, :, 1], mu[:, :, 1])

        new_x = np.concatenate(
            (np.expand_dims(new_x_pet, axis=-1), np.expand_dims(new_x_mr, axis=-1)),
            axis=-1,
        )

        # Z STEP
        new_z = self.z_step(z, new_x, mu)
        new_decoded = self.decoding(new_z)
        new_mean, new_std = self.compute_correction(new_x, new_decoded)
        new_decoded = new_mean + new_decoded * new_std

        # LAGRANGIAN STEP
        new_mu = self.lagrangian_step(new_x, mu, new_decoded)

        # RESIDUALS
        norm_primal_pet, norm_dual_pet = self.compute_residuals(
            new_x[:, :, 0], new_mu[:, :, 0], new_decoded[:, :, 0], decoded[:, :, 0]
        )
        norm_primal_mr, norm_dual_mr = self.compute_residuals(
            new_x[:, :, 1], new_mu[:, :, 1], new_decoded[:, :, 1], decoded[:, :, 1]
        )

        # TAU UPDATE
        self.tau_pet = self.tau_update(norm_primal_pet, norm_dual_pet)
        self.tau_mr = self.tau_update(norm_primal_mr, norm_dual_mr)

        # RHO UPDATE
        update_factor_pet = self.compute_splitted_rho_update_factor(
            norm_primal_pet, norm_dual_pet, self.tau_pet
        )

        update_factor_mr = self.compute_splitted_rho_update_factor(
            norm_primal_mr, norm_dual_mr, self.tau_mr
        )
        self.rho_pet = self.rho_pet * update_factor_pet
        self.rho_mr = self.rho_mr * update_factor_mr

        # new_mu = new_mu / np.array([update_factor_pet, update_factor_mr])
        new_mu[:, :, 0] = new_mu[:, :, 0] / update_factor_pet
        new_mu[:, :, 1] = new_mu[:, :, 1] / update_factor_mr

        stop = False
        if (
            norm_primal_pet <= self.eps_rel
            and norm_dual_pet <= self.eps_rel
            and norm_primal_mr <= self.eps_rel
            and norm_dual_mr <= self.eps_rel
        ):
            stop = True

        return new_x, new_mu, new_z, stop

    def reconstruction(
        self,
        x_pet0,
        x_mr0,
        y_pet,
        y_mr,
        projector_id,
        mri_subsampling,
        step_size,
        nb_steps,
        eps_rel,
    ):
        # Initializations
        self.optimizer = tf.keras.optimizers.Adam(0.05)
        self.nb_steps = nb_steps
        self.step_size = step_size
        self.y_pet = y_pet
        self.y_mr = y_mr
        self.projector_id = projector_id
        self.mri_subsampling = mri_subsampling
        self.eps_rel = eps_rel
        self.N = int(np.prod(x_mr0.shape))
        x_pet0_standardized = (x_pet0 - x_pet0.mean()) / x_pet0.std()
        x_mr0_standardized = (x_mr0 - x_mr0.mean()) / x_mr0.std()
        x0_standardized = np.concatenate(
            (
                np.expand_dims(x_pet0_standardized, axis=-1),
                np.expand_dims(x_mr0_standardized, axis=-1),
            ),
            axis=-1,
        )
        *_, z = self.autoencoder.encoder(np.expand_dims(x0_standardized, axis=0))
        z = tf.Variable(z, trainable=True)

        x = np.concatenate(
            (np.expand_dims(x_pet0, axis=-1), np.expand_dims(x_mr0, axis=-1)), axis=-1
        )
        mu = np.zeros_like(x)
        self.rho_pet = 1 / np.sum(y_pet)
        self.rho_mr = self.rho_pet

        # Beginning of the algorithm
        for _ in range(self.nb_steps):
            x, mu, z, stop = self.reconstruction_step(x, mu, z)
            if stop:
                break

        return x

    def pet_reconstruction_metrics(
        self, x_pet0, ref_pet, ref_mr, y_pet, projector_id, step_size, nb_steps, eps_rel
    ):
        # Initializations
        self.nb_steps = nb_steps
        self.step_size = step_size
        self.y_pet = y_pet
        self.projector_id = projector_id
        x_pet0_normalized = x_pet0 / x_pet0.max()
        ref_mr_normalized = ref_mr / ref_mr.max()
        self.eps_rel = eps_rel
        x0_normalized = np.concatenate(
            (
                np.expand_dims(x_pet0_normalized, axis=-1),
                np.expand_dims(ref_mr_normalized, axis=-1),
            ),
            axis=-1,
        )
        *_, z = self.autoencoder.encoder(np.expand_dims(x0_normalized, axis=0))

        x = np.concatenate(
            (np.expand_dims(x_pet0, axis=-1), np.expand_dims(ref_mr, axis=-1)), axis=-1
        )
        mu = np.zeros_like(x)
        self.rho_pet = 1 / np.sum(y_pet)

        # Initializing metrics
        list_keys = ["nrmse", "ssim", "fidelity", "constraint"]
        points_pet = dict((k, list()) for k in list_keys)

        # Begining of the algo
        for _ in range(nb_steps):
            x, mu, z, stop = self.pet_reconstruction_step(x, mu, z)
            if stop:
                break

            # Plots
            x_pet = x[:, :, 0]
            decoded = self.decoding(z)
            decoded = decoded * self.compute_coeffs(x, decoded)
            points_pet["nrmse"].append(nrmse(ref_pet, x_pet / self.coeffs[0]))
            points_pet["ssim"].append(ssim(x_pet, ref_pet, data_range=x_pet.max()))
            points_pet["fidelity"].append(data_fidelity_pet(x_pet, y_pet, projector_id))
            points_pet["constraint"].append(np.sum((x_pet - decoded[:, :, 0]) ** 2))

        return x[:, :, 0], points_pet

    def reconstruction_metrics(
        self,
        x_pet0,
        x_mr0,
        x_ref,
        y_pet,
        y_mr,
        projector_id,
        mri_subsampling,
        step_size,
        nb_steps,
        eps_rel,
    ):
        # Initializations
        self.optimizer = tf.keras.optimizers.Adam(0.05)
        self.nb_steps = nb_steps
        self.step_size = step_size
        self.y_pet = y_pet
        self.y_mr = y_mr
        self.projector_id = projector_id
        self.mri_subsampling = mri_subsampling
        self.eps_rel = eps_rel
        self.N = int(np.prod(x_mr0.shape))
        x_pet0_standardized = (x_pet0 - x_pet0.mean()) / x_pet0.std()
        x_mr0_standardized = (x_mr0 - x_mr0.mean()) / x_mr0.std()
        x0_standardized = np.concatenate(
            (
                np.expand_dims(x_pet0_standardized, axis=-1),
                np.expand_dims(x_mr0_standardized, axis=-1),
            ),
            axis=-1,
        )
        *_, z = self.autoencoder.encoder(np.expand_dims(x0_standardized, axis=0))
        z = tf.Variable(z, trainable=True)

        x = np.concatenate(
            (np.expand_dims(x_pet0, axis=-1), np.expand_dims(x_mr0, axis=-1)), axis=-1
        )
        mu = np.zeros_like(x)
        self.rho_pet = 1 / np.sum(y_pet)
        self.rho_mr = self.rho_pet

        # Initializing metrics
        list_keys = ["nrmse", "ssim", "fidelity", "constraint"]
        points_pet = dict((k, list()) for k in list_keys)
        points_mr = dict((k, list()) for k in list_keys)

        for _ in range(nb_steps):
            x, mu, z, stop = self.reconstruction_step(x, mu, z)
            if stop:
                break

            # Plots
            x_n = normalize_meanstd(x, (0,1))
            x_pet = x_n[:, :, 0]
            x_mr = x_n[:,:,1]
            ref_pet = x_ref[:,:,0]
            ref_mr = x_ref[:,:,1]

            decoded = self.decoding(z)
            decoded = decoded

            points_pet["nrmse"].append(nrmse(ref_pet, x_pet))
            points_pet["ssim"].append(ssim(x_pet, ref_pet, data_range=x_pet.max()-x_pet.min()))
            points_pet["fidelity"].append(data_fidelity_pet(x_pet, y_pet, projector_id))
            points_pet["constraint"].append(np.sum((x_pet - decoded[:, :, 0]) ** 2))

            points_mr["nrmse"].append(nrmse(ref_mr, x_mr))
            points_mr["ssim"].append(ssim(x_mr, ref_mr, data_range=x_mr.max()-x_mr.min()))
            points_mr["fidelity"].append(data_fidelity_mri(x_mr, y_mr, projector_id))
            points_mr["constraint"].append(np.sum((x_mr - decoded[:, :, 1]) ** 2))


        return x, points_pet, points_mr
