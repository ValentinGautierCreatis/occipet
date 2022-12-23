#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
from occipet.reconstruction import em_step
from scipy.fft import ifft2
from skimage.metrics import structural_similarity as ssim
from .utils import *
import tensorflow as tf
from deep_learning import variational_auto_encoder


class DeepLatentReconstruction():
    def __init__(self, y_pet, y_mri, projector_id,
                 autoencoder: variational_auto_encoder.VariationalAutoEncoder) -> None:
        self.y_pet = y_pet
        self.y_mri = y_mri
        self.projector_id = projector_id
        self.autoencoder = autoencoder
        self.mri_N = 1 #(np.product(y_mri.shape))
        self.xi = 1
        self.tau = 2
        self.tau_max = 100
        self.u = 10


    def decoding(self, z):
        decoded = self.autoencoder.decoder(z).numpy()
        decoded = np.squeeze(decoded, axis=0)
        return decoded


    def pet_step(self, x, pet_decoded, mu):
        _, sensitivity = back_projection(np.ones(self.y_pet.shape,), self.projector_id)
        x_em = em_step(self.y_pet, x, self.projector_id, sensitivity)

        square_root_term = (pet_decoded - mu -
                            (sensitivity/self.rho[0]))**2 + (4*x_em*sensitivity)/self.rho[0]
        # Diff avec la publi, c'est + sqrt et pas -
        return 0.5*(pet_decoded - mu - (sensitivity/self.rho[0]) + np.sqrt(square_root_term))


    def MR_step(self, mr_decoded, mu):

        return (1/(self.rho[1] + self.mri_N)) * (self.mri_N * ifft2(self.y_mri)
                                              + self.rho[1]*(mr_decoded - mu))


    # NOTE: z est reçu sous la forme de Tensor comme ça
    # pas besoin de le reconvertir dans l'algo
    def z_step(self, x, z, mu, coeffs):

        x = tf.Variable(x.reshape((1,) + x.shape), dtype=tf.float32)
        mu = tf.Variable(mu.reshape((1,) + mu.shape), dtype=tf.float32)
        with tf.GradientTape() as tape:

            tape.watch(z)
            # careful with the shape z and decoded
            decoded = self.autoencoder.decoder(z) * coeffs
            new_image = (x + mu) - decoded
            squared = tf.math.square(new_image)
            # squared = tf.math.multiply(decoded, decoded)
            # product = tf.math.multiply(x + mu, decoded)

        squared_gradient = tape.gradient(squared, z)
        # product_gradient = tape.gradient(product, z)


        update_term = (squared_gradient)

        return z - self.step_size*update_term


    def lagragian_step(self, x, z, mu, coeffs):
        decoded = self.autoencoder.decoder(z).numpy() * coeffs
        decoded = decoded.reshape(decoded.shape[1:])
        return mu + x - decoded


    def compute_coeffs(self, x, decoded):
        return x.mean(axis=(0,1))/decoded.mean(axis=(0,1))


    def reconstruct(self, x0, mu0, rho, step_size, nb_iterations):
        self.rho = np.array(rho)
        self.step_size = step_size

        *_, z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0

        for _ in range(nb_iterations):
            decoded = self.decoding(z)
            coeffs = self.compute_coeffs(x, decoded)

            decoded = decoded * coeffs
            pet_decoded = decoded[:,:,0]
            mr_decoded = decoded[:,:,1]

            x_pet = x[:, :, 0]
            new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:,:,0])
            new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))

            new_x_mr = self.MR_step(mr_decoded, mu[:,:,1])
            new_x_mr = abs(new_x_mr.reshape(new_x_mr.shape + (1,)))
            x = np.concatenate((new_x_pet, new_x_mr), axis = 2)

            z = self.z_step(x, z, mu, coeffs)

            mu = self.lagragian_step(x, z, mu, coeffs)

        return x


    def eval(self, x0, mu0, rho, step_size, nb_iterations, ref_pet, ref_mr):
        self.rho = np.array(rho)
        self.step_size = step_size

        *_, z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0

        list_keys = ["mse", "ssim"]
        points_pet = dict((k, list()) for k in list_keys)
        points_mr = dict((k, list()) for k in list_keys)

        for _ in range(nb_iterations):
            decoded = self.decoding(z)
            coeffs = self.compute_coeffs(x, decoded)

            decoded = decoded * coeffs
            pet_decoded = decoded[:,:,0]
            mr_decoded = decoded[:,:,1]

            x_pet = x[:, :, 0]
            new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:,:,0])
            new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))

            new_x_mr = self.MR_step(mr_decoded, mu[:,:,1])
            new_x_mr = abs(new_x_mr.reshape(new_x_mr.shape + (1,)))
            x = np.concatenate((new_x_pet, new_x_mr), axis = 2)

            z = self.z_step(x, z, mu, coeffs)

            mu = self.lagragian_step(x, z, mu, coeffs)

            normalised_pet = x[:,:,0] / x[:,:,0].max()
            normalised_mr = x[:,:,1] / x[:,:,1].max()

            points_pet["mse"].append(mse(normalised_pet, ref_pet))
            points_pet["ssim"].append(ssim(normalised_pet, ref_pet,
                                           data_range=normalised_pet.max()))
            points_mr["mse"].append(mse(normalised_mr, ref_mr))
            points_mr["ssim"].append(ssim(normalised_mr, ref_mr,
                                          data_range=normalised_mr.max()))

        return x, points_pet, points_mr


    def reconstruction_pet_step(self, x, z, mu, ref_mr):

        decoded = self.decoding(z)
        self.coeffs = self.compute_coeffs(x, decoded)

        decoded = decoded * self.coeffs
        pet_decoded = decoded[:,:,0]
        # mr_decoded = decoded[:,:,1]

        x_pet = x[:, :, 0]
        new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:,:,0])
        new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))

        x = np.concatenate((new_x_pet, ref_mr), axis = 2)

        z = self.z_step(x, z, mu, self.coeffs)

        mu = self.lagragian_step(x, z, mu, self.coeffs)
    
        return x, z, mu


    def reconstruction_pet(self, xpet0, ref_mr, mu0, rho, step_size, nb_iterations):
        self.rho = np.array(rho)
        self.step_size = step_size

        x0 = np.concatenate((xpet0, ref_mr), axis=2)
        *_, z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0

        for _ in range(nb_iterations):
            x, z, mu = self.reconstruction_pet_step(x, z, mu, ref_mr)
        return x,z


    def reconstruction_pet_eval(self, xpet0, ref_pet, ref_mr, mu0, rho, step_size, nb_iterations):
        self.rho = np.array(rho)
        self.step_size = step_size

        x0 = np.concatenate((xpet0, ref_mr), axis=2)
        *_, z = self.autoencoder.encoder(np.expand_dims(x0, axis=0))
        x = x0
        mu = mu0

        list_keys = ["mse", "ssim", "fidelity", "constraint"]
        points_pet = dict((k, list()) for k in list_keys)

        for _ in range(nb_iterations):
            old_z = np.array(z, copy=True)

            x, z, mu = self.reconstruction_pet_step(x, z, mu, ref_mr)

            primal_residual = self.compute_primal_residual(x[:,:,0], z, self.coeffs[0])
            dual_residual = self.compute_dual_residual(z, old_z, mu, self.coeffs[0])

            self.tau = self.tau_update(primal_residual, dual_residual)

            self.rho[0] = self.rho_update(self.rho[0], primal_residual, dual_residual)
            # print(self.rho)
            normalised_pet = x[:,:,0] #/ x[:,:,0].max()

            points_pet["mse"].append(mse(normalised_pet, np.squeeze(ref_pet, axis=-1)))
            points_pet["ssim"].append(ssim(normalised_pet, np.squeeze(ref_pet, axis=-1),
                                           data_range=normalised_pet.max()))
            points_pet["fidelity"].append(data_fidelity_pet(x[:,:,0], self.y_pet, self.projector_id))
            points_pet["constraint"].append(np.sum( ((x - self.decoding(z))**2)[:,:,0] ))

            pet_dec = self.decoding(z)[:,:,0]
            plt.imshow(x[:,:,0])
            plt.colorbar()
            plt.show()

        return x,z,points_pet


    def compute_primal_residual(self, x, z, coeff):
        decoded_z = coeff * self.decoding(z)[:,:,0]
        norm = np.max([np.linalg.norm(x), np.linalg.norm(decoded_z)])

        return (x - decoded_z) / norm


    def compute_dual_residual(self, z_k1, z_k, mu, coeff):
        decoded_z_k1 = self.decoding(z_k1)[:,:,0]
        decoded_z_k = self.decoding(z_k)[:,:,0]
        norm = np.linalg.norm(mu[:,:,0])

        return coeff * (decoded_z_k1 - decoded_z_k) / norm


    def rho_update(self, rho, primal_residual, dual_residual):
        norm_primal = np.linalg.norm(primal_residual)
        norm_dual = np.linalg.norm(dual_residual)

        # print(f"norm primal: {norm_primal}")
        # print(f"norm dual: {norm_dual}")

        if norm_primal > self.xi * self.u * norm_dual:
            return self.tau * rho

        elif norm_dual > (self.u/self.xi) * norm_primal:
            return rho / self.tau

        else:
            return rho


    def tau_update(self, primal_residual, dual_residual):
        norm_primal = np.linalg.norm(primal_residual)
        norm_dual = np.linalg.norm(dual_residual)
        coeff = np.sqrt(norm_primal/(norm_dual*self.xi))

        if 1 <= coeff < self.tau_max:
            return coeff

        elif (1/self.tau_max) < coeff < 1:
            return 1/coeff

        else:
            return self.tau_max


    def test_reconstruction(self, xpet0, ref_mr, mu0, rho, step_size, nb_iterations):
        self.rho = np.array(rho)
        self.step_size = step_size

        x0 = np.concatenate((xpet0, ref_mr), axis=2)
        *_, z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0

        for _ in range(nb_iterations):
            decoded = self.decoding(z)
            coeffs = self.compute_coeffs(x, decoded)

            decoded = decoded * coeffs
            pet_decoded = decoded[:,:,0]

            x_pet = x[:, :, 0]
            new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:,:,0])
            new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))

            x = np.concatenate((new_x_pet, ref_mr), axis = 2)

            old_z = np.array(z, copy=True)
            z = self.z_step(x, z, mu, coeffs)

            mu = self.lagragian_step(x, z, mu, coeffs)

            primal_residual = self.compute_primal_residual(x, z, coeffs[0])
            dual_residual = self.compute_dual_residual(z, old_z, mu, coeffs[0])

            self.tau = self.tau_update(primal_residual, dual_residual)

            self.rho = self.rho_update(self.rho, primal_residual, dual_residual)

        return x,z


class DLRtest(DeepLatentReconstruction):
    def __init__(self, y_pet, y_mri, projector_id,
                 autoencoder: variational_auto_encoder.VariationalAutoEncoder) -> None:
        super().__init__(y_pet, y_mri, projector_id, autoencoder)


    def reconstruct(self, x0, mu0, rho, step_size, nb_iterations, ref_pet):
        z_pet_quality = []
        self.rho = np.array(rho)
        self.step_size = step_size

        *_, z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0

        for _ in range(nb_iterations):
            decoded = self.decoding(z)
            coeffs = self.compute_coeffs(x, decoded)

            decoded = decoded * coeffs
            pet_decoded = decoded[:,:,0]
            mr_decoded = decoded[:,:,1]

            x_pet = x[:, :, 0]
            new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:,:,0])
            new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))

            new_x_mr = self.MR_step(mr_decoded, mu[:,:,1])
            new_x_mr = abs(new_x_mr.reshape(new_x_mr.shape + (1,)))
            x = np.concatenate((new_x_pet, new_x_mr), axis = 2)

            z = self.z_step(x, z, mu, coeffs)

            mu = self.lagragian_step(x, z, mu, coeffs)

            pet_ = self.autoencoder.decoder(z).numpy()[:,:,:,0].reshape(x.shape[:-1])
            pet_error = np.sum(np.square(pet_ - ref_pet))

            z_pet_quality.append(pet_error)
        return x,z, z_pet_quality


    def reconstruction_pet_step_ref(self, x, mu, ref_pet):
        decoded = ref_pet
        coeffs = self.compute_coeffs(x, decoded)

        decoded = decoded * coeffs
        pet_decoded = decoded
        # mr_decoded = decoded[:,:,1]

        x_pet = np.array(x, copy=True)
        new_x_pet = self.pet_step(x_pet, pet_decoded, mu)

        x = new_x_pet

        mu = mu + x - decoded

        primal_residual = x - ref_pet
        dual_residual = 0 * x

        self.rho = self.rho_update(self.rho, primal_residual, dual_residual)

        return x, mu


    def reconstruct_ref(self, xpet0, ref_pet, mu0, rho, step_size, nb_iterations):
        self.rho = np.array(rho)
        self.step_size = step_size

        x0 = xpet0
        x = np.array(x0, copy=True)
        mu = mu0

        for _ in range(nb_iterations):
            x, mu = self.reconstruction_pet_step_ref(x, mu, ref_pet)
        return x
