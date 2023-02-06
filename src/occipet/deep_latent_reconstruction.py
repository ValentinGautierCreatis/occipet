#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
from occipet.reconstruction import em_step
from skimage.metrics import structural_similarity as ssim
from .utils import *
import tensorflow as tf
from deep_learning import variational_auto_encoder


class DeepLatentReconstruction:
    def __init__(
        self, autoencoder: variational_auto_encoder.VariationalAutoEncoder
    ) -> None:
        self.autoencoder = autoencoder
        self.xi = 1
        self.u = 10
        self.tau = 1
        self.tau_max = 100
        self.coeffs = np.array([1, 1])
        self.rho_pet: float
        self.rho_mr: float
        self.y_pet: np.ndarray
        self.y_mr: np.ndarray
        self.projector_id: int

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

    def z_step(self, z, x, mu):
        with tf.GradientTape() as tape:
            tape.watch(z)
            decoded = self.autoencoder.decoder(z) * self.coeffs
            new_image = (x + mu) - decoded
            squared = tf.math.square(new_image)
        update_term = tape.gradient(squared, z)

        return z - self.step_size * update_term

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

    def compute_coeffs(self, x, decoded):
        return x.mean(axis=(0, 1)) / decoded.mean(axis=(0, 1))

    def decoding(self, z):
        return np.squeeze(self.autoencoder.decoder(z).numpy(), axis=0)

    def pet_reconstruction_step(self, x, mu, z):
        decoded = self.decoding(z)
        self.coeffs = self.compute_coeffs(x, decoded)
        decoded = self.coeffs * decoded
        # print(self.rho_pet)
        # plt.imshow(x[:,:,1])
        # plt.colorbar()
        # plt.show()
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
        new_coeff = self.compute_coeffs(x, new_decoded)
        new_decoded = new_coeff * new_decoded

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
        new_mu[:, :, 0] = new_mu[:, :, 0] / update_factor

        stop = False
        if norm_primal <= self.eps_rel and norm_dual <= self.eps_rel:
            stop = True

        return new_x, new_mu, new_z, stop

    # Input images with 2 shapes
    def pet_reconstruction(
        self, x_pet0, ref_mr, y_pet, projector_id, step_size, nb_steps, eps_rel
    ):
        # Initializations
        self.nb_steps = nb_steps
        self.step_size = step_size
        self.y_pet = y_pet
        self.projector_id = projector_id
        self.eps_rel = eps_rel
        x_pet0_normalized = x_pet0 / x_pet0.max()
        ref_mr_normalized = ref_mr / ref_mr.max()
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

        # Begining of the algo
        for _ in range(self.nb_steps):
            x, mu, z, stop = self.pet_reconstruction_step(x, mu, z)
            if stop:
                break

        return x  # [:,:,0]

    def pet_reconstruction_metrics(
        self, x_pet0, ref_pet, ref_mr, y_pet, projector_id, step_size
    ):
        # Initializations
        nb_steps = 40
        self.step_size = step_size
        self.y_pet = y_pet
        self.projector_id = projector_id
        x_pet0_normalized = x_pet0 / x_pet0.max()
        ref_mr_normalized = ref_mr / ref_mr.max()
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
        list_keys = ["mse", "ssim", "fidelity", "constraint"]
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
            points_pet["mse"].append(mse(ref_pet, x_pet / self.coeffs[0]))
            points_pet["ssim"].append(ssim(x_pet, ref_pet, data_range=x_pet.max()))
            points_pet["fidelity"].append(data_fidelity_pet(x_pet, y_pet, projector_id))
            points_pet["constraint"].append(np.sum((x_pet - decoded[:, :, 0]) ** 2))

        return x[:, :, 0], points_pet
