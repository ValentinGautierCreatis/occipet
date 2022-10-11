#!/usr/bin/env python3

import numpy as np

from occipet.reconstruction import em_step
from scipy.fft import ifft2
from math import sqrt
from .utils import *
import tensorflow as tf
from deep_learning import variational_auto_encoder


class DeepLatentReconstruction():
    def __init__(self, y_pet, y_mri, rho, step_size, projector_id,
                 autoencoder: variational_auto_encoder.VariationalAutoEncoder,
                 ) -> None:
        self.y_pet = y_pet
        self.y_mri = y_mri
        self.rho = rho
        self.step_size= step_size
        self.projector_id = projector_id
        self.autoencoder = autoencoder
        self.mri_N = 1 #(np.product(y_mri.shape))


    def decoding(self, z):
        decoded = self.autoencoder.decoder(z).numpy()
        decoded = np.squeeze(decoded, axis=0)
        return decoded


    def pet_step(self, x, pet_decoded, mu):
        _, sensitivity = back_projection(np.ones(self.y_pet.shape,), self.projector_id)
        x_em = em_step(self.y_pet, x, self.projector_id, sensitivity)

        square_root_term = (pet_decoded - mu -
                            (sensitivity/self.rho))**2 + (4*x_em*sensitivity)/self.rho
        # Diff avec la publi, c'est + sqrt et pas -
        return 0.5*(pet_decoded - mu - (sensitivity/self.rho) + np.sqrt(square_root_term))


    def MR_step(self, mr_decoded, mu):

        return (1/(self.rho + self.mri_N)) * (self.mri_N * ifft2(self.y_mri)
                                              + self.rho*(mr_decoded - mu))


    # NOTE: z est reçu sous la forme de Tensor comme ça
    # pas besoin de le reconvertir dans l'algo
    def z_step(self, x, z, mu):

        x = tf.Variable(x.reshape((1,) + x.shape), dtype=tf.float32)
        mu = tf.Variable(mu.reshape((1,) + mu.shape), dtype=tf.float32)
        with tf.GradientTape() as tape:

            tape.watch(z)
            # careful with the shape z and decoded
            decoded = self.autoencoder.decoder(z)
            new_image = (x + mu) - decoded
            squared = tf.math.square(new_image)
            # squared = tf.math.multiply(decoded, decoded)
            # product = tf.math.multiply(x + mu, decoded)

        squared_gradient = tape.gradient(squared, z)
        # product_gradient = tape.gradient(product, z)


        update_term = self.rho/2 * (squared_gradient)

        return z - self.step_size*update_term


    def lagragian_step(self, x, z, mu):
        decoded = self.autoencoder.decoder(z).numpy()
        decoded = decoded.reshape(decoded.shape[1:])
        return mu + x - decoded


    def reconstruct(self, x0, mu0, nb_iterations):
        *_, z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0
        for _ in range(nb_iterations):

            decoded = self.decoding(z)
            pet_decoded = decoded[:,:,0]
            mr_decoded = decoded[:,:,1]
            x_pet = x[:, :, 0]
            new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:,:,0])
            new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))
            new_x_mr = self.MR_step(mr_decoded, mu[:,:,1])
            new_x_mr = abs(new_x_mr.reshape(new_x_mr.shape + (1,)))
            x = np.concatenate((new_x_pet, new_x_mr), axis = 2)

            z = self.z_step(x, z, mu)

            mu = self.lagragian_step(x, z, mu)

        return x



    def z_step_test(self, x, z, mu, coeffs):

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


        update_term = self.rho/2 * (squared_gradient)

        return z - self.step_size*update_term


    def lagragian_step_test(self, x, z, mu, coeffs):
        decoded = self.autoencoder.decoder(z).numpy() * coeffs
        decoded = decoded.reshape(decoded.shape[1:])
        # decoded = 1000 * np.ones_like(x) # <============== Test only
        return mu + x - decoded


    def reconstruct_test(self, x0, mu0, coeff, nb_iterations, ref_pet):
        *_, z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0
        z_pet_quality = []
        coeffs = np.array([coeff, 1])
        # pet_decoded = 1000*np.ones_like(x[:,:,0])
        # mr_decoded = 1000*np.ones_like(x[:,:,1])
        for _ in range(nb_iterations):

            decoded = self.decoding(z) * coeffs
            pet_decoded = decoded[:,:,0]
            mr_decoded = decoded[:,:,1]


            x_pet = x[:, :, 0]
            new_x_pet = self.pet_step(x_pet, pet_decoded, mu[:,:,0])
            new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))
            new_x_mr = self.MR_step(mr_decoded, mu[:,:,1])
            new_x_mr = abs(new_x_mr.reshape(new_x_mr.shape + (1,)))
            x = np.concatenate((new_x_pet, new_x_mr), axis = 2)

            # z = self.z_step_test(x, z, mu, coeffs)

            mu = self.lagragian_step_test(x, z, mu, coeffs)

            pet_ = self.autoencoder.decoder(z).numpy()[:,:,:,0].reshape(x.shape[:-1])
            pet_error = np.sum(np.square(pet_ - ref_pet))

            z_pet_quality.append(pet_error)
        return x,z, z_pet_quality
