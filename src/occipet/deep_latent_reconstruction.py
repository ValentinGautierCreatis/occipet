#!/usr/bin/env python3

import numpy as np

from occipet.reconstruction import em_step
from scipy.fft import ifft2
from .utils import *
import tensorflow as tf
from . import VariationalAutoEncoder

# TODO: Mettre les Decoder(z) au bon format (tel quel ce sont des tensor shape (1, w, h, 2))
class DeepLatentReconstruction():
    def __init__(self, y_pet, y_mri, rho, step_size, projector_id,
                 autoencoder: VariationalAutoEncoder.VariationalAutoEncoder,
                 S) -> None:
        self.y_pet = y_pet
        self.y_mri = y_mri
        self.rho = rho
        self.step_size= step_size
        self.projector_id = projector_id
        self.autoencoder = autoencoder
        self.mri_N = np.product(y_mri.shape)
        self.S = S


    def pet_step(self, x, z, mu):

        _, sensitivity = back_projection(np.ones(self.y_pet.shape,), self.projector_id)
        x_em = em_step(self.y_pet, x, self.projector_id, sensitivity)

        square_root_term = (self.autoencoder.decoder(z) - mu -
                            (sensitivity/self.rho))**2 + (4*x_em*sensitivity)/self.rho
        return 0.5*(self.autoencoder.decoder(z) - mu - (sensitivity/self.rho) - np.sqrt(square_root_term))


    def MR_step(self, z, mu):

        return (1/(self.rho + self.mri_N)) * (self.mri_N * ifft2(self.y_mri)
                                              + self.rho*(self.autoencoder.decoder(z)) - mu)


    # NOTE: z est reçu sous la forme de Tensor comme ça
    # pas besoin de le reconvertir dans l'algo
    def z_step(self, x, z, mu):

        x = tf.Variable(x.reshape((1,) + x.shape))
        mu = tf.Variable(mu.reshape((1,) + mu.shape))
        with tf.GradientTape() as tape:

            tape.watch(z)
            # careful with the shape z and decoded
            decoded = self.autoencoder.decoder(z)
            new_image = (x + mu) - decoded
            squared = tf.math.multiply(new_image, new_image)
            # squared = tf.math.multiply(decoded, decoded)
            # product = tf.math.multiply(x + mu, decoded)

        squared_gradient = tape.gradient(squared, z)
        # product_gradient = tape.gradient(product, z)


        update_term = self.rho * (squared_gradient)

        return z - self.S*update_term



    def lagragian_step(self, x, z, mu):
        return mu + x - self.autoencoder.decoder(z)


    def reconstruct(self, x0, mu0, nb_iterations):
        z = self.autoencoder.encoder(x0.reshape((1,) + x0.shape))
        x = x0
        mu = mu0
        for _ in range(nb_iterations):

            x_pet = x0[:, :, 0]
            new_x_pet = self.pet_step(x_pet, z, mu)
            new_x_pet = new_x_pet.reshape(new_x_pet.shape + (1,))
            new_x_mr = self.MR_step(z, mu)
            new_x_mr = new_x_mr.reshape(new_x_mr.shape + (1,))
            x = np.concatenate((new_x_pet, new_x_mr), axis = 2)

            z = self.z_step(x, z, mu)

            mu = self.lagragian_step(x, z, mu)

        return x
