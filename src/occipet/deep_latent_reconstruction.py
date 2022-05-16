#!/usr/bin/env python3

import numpy as np

from occipet.reconstruction import em_step
from scipy.fft import ifft2
from .utils import *
import astra
import tensorflow as tf
from . import VariationalAutoEncoder

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

        sensitivity = back_projection(np.ones(self.y_pet.shape,), self.projector_id)
        x_em = em_step(self.y_pet, x, self.projector_id, sensitivity)

        square_root_term = (self.autoencoder.decoder(z) - mu -
                            (sensitivity/self.rho))**2 + (4*x_em*sensitivity)/self.rho
        return 0.5*(self.autoencoder.decoder(z) - mu - (sensitivity/self.rho) - np.sqrt(square_root_term))


    def MR_step(self, z, mu):

        return (1/(self.rho + self.mri_N)) * (self.mri_N * ifft2(self.y_mri)
                                              + self.rho*(self.autoencoder.decoder(z)) - mu)


    # TODO: les dimensions sont complètements fausses sur z
    # c'est pas une image. Généraliser les dimensions de z ?
    def z_step(self, x, z, mu):

        h, w = z.shape
        z_tf = tf.Variable(z.reshape((1, h, w, 1)))
        with tf.GradientTape() as tape:

            tape.watch(z_tf)
            # careful with the shape z and decoded
            decoded = self.autoencoder.decoder(z_tf)

        jacobian = tape.jacobian(decoded, z_tf).numpy()
        jacobian = jacobian.reshape(jacobian.shape[1], jacobian.shape[2],
                                    jacobian.shape[5], jacobian.shape[6])


        lagrangian_multiplier = self.rho * (x + mu - decoded)

        product_jacobian = np.empty_like(jacobian)
        for i in range(h):
            for j in range(w):
                product_jacobian[:, :, i, j] = jacobian[:, :, i, j] * lagrangian_multiplier

        update_term = np.sum(product_jacobian, axis=(2,3))

        return z - self.S*update_term



    def lagragian_step(self, x, z, mu):
        return mu + x - self.autoencoder.decoder(z)


    def reconstruct(self, x0, mu0):
        z0 =
