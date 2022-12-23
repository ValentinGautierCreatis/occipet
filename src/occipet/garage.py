#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
from occipet.reconstruction import em_step
from .utils import *
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

def dlr(y_pet, projector_id, xpet0, ref_pet, ref_mr, mu0, model, rho, step_size, nb_iterations):
    xpet0_normalized = xpet0/xpet0.max()
    x0 = np.concatenate((xpet0, ref_mr), axis = -1)
    x0_normalized = np.concatenate((xpet0_normalized, ref_mr), axis = -1)
    *_, z = model.encoder(np.expand_dims(x0_normalized, axis=0))
    x = x0
    mu = mu0
    rho = np.array(rho)

    xi = 1
    u = 10
    tau = 1
    tau_max = 100

    list_keys = ["mse", "ssim", "fidelity", "constraint"]
    points_pet = dict((k, list()) for k in list_keys)

    points_pet["mse"].append(mse(np.squeeze(xpet0, axis=-1), np.squeeze(ref_pet, axis=-1)))
    for _ in range(nb_iterations):
        old_z = np.array(z, copy=True)
        decoded = np.squeeze(model.decoder(z).numpy(), axis=0)
        coeffs = x.mean(axis=(0,1))/decoded.mean(axis=(0,1))
        decoded = coeffs * decoded
        pet_decoded = decoded[:,:,0]

        # print(rho)
        plt.imshow(decoded[:,:,0]/coeffs[0])
        plt.colorbar()
        plt.show()

        # PET STEP =========================================
        x_pet = x[:,:,0]
        _, sensitivity = back_projection(np.ones(y_pet.shape), projector_id)
        x_em = em_step(y_pet, x_pet, projector_id, sensitivity)
        square_root_term = ( (pet_decoded - mu[:,:,0] - (sensitivity/rho))**2
                             + (4 * x_em * sensitivity)/rho)
        new_xpet = 0.5 * (pet_decoded - mu[:,:,0] - (sensitivity/rho) + np.sqrt(square_root_term))

        x = np.concatenate((np.expand_dims(new_xpet, axis=-1), ref_mr), axis=-1)

        # Z STEP ===========================================
        # tensor_x = tf.Variable(np.expand_dims(x, axis=0))
        # tensor_mu = tf.Variable(np.expand_dims(mu, axis=0))
        with tf.GradientTape() as tape:
            tape.watch(z)
            decoded = model.decoder(z) * coeffs
            new_image = (x + mu) - decoded
            squared = tf.math.square(new_image)
        update_term = tape.gradient(squared, z)
        z = z - step_size * update_term

        # LAGRAGIAN STEP
        decoded = np.squeeze(model.decoder(z).numpy(), axis=0)
        coeffs = x.mean(axis=(0,1))/decoded.mean(axis=(0,1))
        decoded = coeffs * decoded
        mu = mu + x - decoded

        # RESIDUALS
        #
        normalisation_primal = np.max([np.linalg.norm(x[:,:,0]), np.linalg.norm(decoded[:,:0])])
        norm_primal = np.linalg.norm( (x - decoded)[:,:,0]/normalisation_primal )

        decoded_zk = np.squeeze(model.decoder(old_z).numpy(), axis=0)
        coeffs_prev = x.mean(axis=(0,1))/decoded_zk.mean(axis=(0,1))
        decoded_zk = coeffs_prev * decoded_zk
        normalisation_dual = np.linalg.norm(mu[:,:,0])
        norm_dual = np.linalg.norm((decoded - decoded_zk)[:,:,0]/normalisation_dual)

        #TAU UPDATE
        update_coeff = np.sqrt(norm_primal/(norm_dual*xi))
        if 1 <= update_coeff < tau_max:
            tau = update_coeff
        elif (1/tau_max) < update_coeff < 1:
            tau = 1/update_coeff
        else:
            tau = tau_max

        #RHO UPDATE
        if norm_primal > xi * u * norm_dual:
            rho = tau * rho
        elif norm_dual > (u/xi) * norm_primal:
            rho = rho/tau


        #PLOTS
        points_pet["mse"].append(mse(new_xpet, np.squeeze(ref_pet, axis=-1)))
        points_pet["ssim"].append(ssim(new_xpet, np.squeeze(ref_pet, axis=-1),
                                    data_range=new_xpet.max()))
        points_pet["fidelity"].append(data_fidelity_pet(x[:,:,0], y_pet, projector_id))
        points_pet["constraint"].append(np.sum((x[:,:,0] - decoded[:,:,0])**2))

    return x, points_pet
