#!/usr/bin/env python3
import numpy as np
import occipet.load_data as load_data
from occipet import reconstruction
from occipet import utils

import matplotlib.pyplot as plt
import random
from occipet.reconstruction import em_step
from .utils import *
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from deep_learning.variational_auto_encoder import VariationalAutoEncoder


# USED ONLY FOR TESTING, DEPRECATED


def dlr(
    y_pet,
    projector_id,
    xpet0,
    ref_pet,
    ref_mr,
    mu0,
    model,
    rho,
    step_size,
    nb_iterations,
):
    xpet0_normalized = xpet0 / xpet0.max()
    x0 = np.concatenate((xpet0, ref_mr), axis=-1)
    x0_normalized = np.concatenate((xpet0_normalized, ref_mr), axis=-1)
    *_, z = model.encoder(np.expand_dims(x0_normalized, axis=0))
    x = x0
    mu = mu0
    rho = np.array(rho)

    xi = 1
    u = 10
    tau = 1
    tau_max = 100
    # rho_max = 10

    list_keys = ["mse", "ssim", "fidelity", "constraint"]
    points_pet = dict((k, list()) for k in list_keys)

    points_pet["mse"].append(
        nrmse(np.squeeze(ref_pet, axis=-1), np.squeeze(xpet0, axis=-1))
    )
    for _ in range(nb_iterations):
        old_z = np.array(z, copy=True)
        decoded = np.squeeze(model.decoder(z).numpy(), axis=0)
        coeffs = x.mean(axis=(0, 1)) / decoded.mean(axis=(0, 1))
        decoded = coeffs * decoded
        pet_decoded = decoded[:, :, 0]

        # print(rho)
        # plt.imshow(x[:, :, 0])
        # plt.imshow(decoded[:, :, 0] / coeffs[0])
        # plt.colorbar()
        # plt.show()

        # PET STEP =========================================
        x_pet = x[:, :, 0]
        _, sensitivity = back_projection(np.ones(y_pet.shape), projector_id)
        x_em = em_step(y_pet, x_pet, projector_id, sensitivity)
        square_root_term = (pet_decoded - mu[:, :, 0] - (sensitivity / rho)) ** 2 + (
            4 * x_em * sensitivity
        ) / rho
        new_xpet = 0.5 * (
            pet_decoded - mu[:, :, 0] - (sensitivity / rho) + np.sqrt(square_root_term)
        )

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
        coeffs = x.mean(axis=(0, 1)) / decoded.mean(axis=(0, 1))
        decoded = coeffs * decoded
        mu = mu + x - decoded

        # RESIDUALS
        #
        normalisation_primal = np.max(
            [np.linalg.norm(x[:, :, 0]), np.linalg.norm(decoded[:, :, 0])]
        )
        norm_primal = np.linalg.norm((x - decoded)[:, :, 0] / normalisation_primal)

        decoded_zk = np.squeeze(model.decoder(old_z).numpy(), axis=0)
        coeffs_prev = x.mean(axis=(0, 1)) / decoded_zk.mean(axis=(0, 1))
        decoded_zk = coeffs_prev * decoded_zk
        normalisation_dual = np.linalg.norm(mu[:, :, 0])
        norm_dual = np.linalg.norm((decoded - decoded_zk)[:, :, 0] / normalisation_dual)

        # TAU UPDATE
        update_coeff = np.sqrt(norm_primal / (norm_dual * xi))
        if 1 <= update_coeff < tau_max:
            tau = update_coeff
        elif (1 / tau_max) < update_coeff < 1:
            tau = 1 / update_coeff
        else:
            tau = tau_max

        # RHO UPDATE
        update_factor = 1
        if norm_primal > xi * u * norm_dual:
            # rho = min(tau * rho, rho_max)
            rho = rho * tau
            update_factor = tau
        elif norm_dual > (u / xi) * norm_primal:
            # rho = min(rho/tau, rho_max)
            rho = rho / tau
            update_factor = 1 / tau

        mu[:, :, 0] = mu[:, :, 0] / update_factor

        # STOPPING CRITERION
        eps_rel = 0.02
        if norm_primal <= eps_rel and norm_dual <= eps_rel:
            break

        # PLOTS
        # print(np.mean(np.squeeze(ref_pet, axis=-1)))
        points_pet["mse"].append(
            nrmse(np.squeeze(ref_pet, axis=-1), new_xpet / coeffs[0])
        )
        points_pet["ssim"].append(
            ssim(new_xpet, np.squeeze(ref_pet, axis=-1), data_range=new_xpet.max())
        )
        points_pet["fidelity"].append(
            data_fidelity_pet(x[:, :, 0], y_pet, projector_id)
        )
        points_pet["constraint"].append(np.sum((x[:, :, 0] - decoded[:, :, 0]) ** 2))

    return x, points_pet


def evaluate():
    random.seed(10)
    path_to_data = "/home/gautier/data/all_patients_train.npy"
    path_to_model = "/home/gautier/Modèles/bimodal_v2/"
    index = 670

    data = np.load(path_to_data)
    model = VariationalAutoEncoder((256, 256, 2), 64)
    model.load_weights(path_to_model)

    nb_iterations = 40
    nb_init_steps = 10
    mlem_iterations = 15
    nb_photons = 2 * 10**5
    ####### TO TUNE ######
    S = 0.001
    ######################

    pet_image = data[index, :, :, 0]
    mri_image = data[index, :, :, 1]
    mu_2d = np.zeros(np.expand_dims(pet_image, axis=-1).shape)

    y_pet, projector_id = load_data.generate_pet_data_from_image(
        pet_image, 0.01, 60, nb_photons
    )
    # y_pet, projector_id = load_data.generate_pet_data_from_image(pet_image, 0.5, 200, 12**6)

    y_mr, sigma = load_data.generate_t1_mr_data_from_image(mri_image, 0.03)

    x_init_pet = reconstruction.MLEM(
        y_pet, pet_image.shape, nb_init_steps, projector_id
    )
    x_init_pet = x_init_pet.reshape(x_init_pet.shape + (1,))

    x_init_mr = abs(ifft2(y_mr))
    x_init_mr = x_init_mr.reshape(x_init_mr.shape + (1,)) / x_init_mr.max()

    # test_rho = [1/np.sum(x_init_pet), 0.2]
    test_rho = [1 / np.sum(y_pet), 0.2]

    # coeff = x_init_pet.mean()/data[index,:,:,0].mean()
    x, points_pet = dlr(
        y_pet,
        projector_id,
        x_init_pet,
        data[index, :, :, :1],
        data[index, :, :, 1:],
        mu_2d,
        model,
        test_rho[0],
        S * 10,
        nb_iterations,
    )
    ref_mlem = reconstruction.MLEM(
        y_pet, x_init_pet.shape[:2], mlem_iterations, projector_id
    )
    mse_mlem = utils.nrmse(
        data[index, :, :, 0], ref_mlem / ref_mlem.mean() * data[index, :, :, 0].mean()
    )

    return x, points_pet, mse_mlem


def evaluation(nb_photons, S):
    indexes = [670, 25, 74, 127, 294, 425, 696, 731, 521]
    # indexes = [0, 12, 15, 22, 24, 25, 26, 30, 32, 38, 61, 68]
    keys = ["mse", "ssim"]
    metrics_dlr = dict((k, list()) for k in keys)
    metrics_mlem = dict((k, list()) for k in keys)

    path_to_data = "/home/gautier/data/all_patients_train.npy"
    path_to_model = "/home/gautier/Modèles/bimodal_v2/"
    index = 25

    data = np.load(path_to_data)
    model = VariationalAutoEncoder((256, 256, 2), 64)
    model.load_weights(path_to_model)

    nb_iterations = 40
    nb_init_steps = 10
    mlem_iterations = 15

    for index in indexes:
        pet_image = data[index, :, :, 0]
        mri_image = data[index, :, :, 1]
        mu_2d = np.zeros(np.expand_dims(pet_image, axis=-1).shape)

        y_pet, projector_id = load_data.generate_pet_data_from_image(
            pet_image, 0.01, 60, nb_photons
        )
        y_mr, sigma = load_data.generate_t1_mr_data_from_image(mri_image, 0.03)

        x_init_pet = reconstruction.MLEM(
            y_pet, pet_image.shape, nb_init_steps, projector_id
        )
        x_init_pet = x_init_pet.reshape(x_init_pet.shape + (1,))

        x_init_mr = abs(ifft2(y_mr))
        x_init_mr = x_init_mr.reshape(x_init_mr.shape + (1,)) / x_init_mr.max()

        test_rho = [1 / np.sum(x_init_pet), 0.2]

        # coeff = x_init_pet.mean()/data[index,:,:,0].mean()
        x, p = dlr(
            y_pet,
            projector_id,
            x_init_pet,
            data[index, :, :, :1],
            data[index, :, :, 1:],
            mu_2d,
            model,
            test_rho[0],
            S * 10,
            nb_iterations,
        )
        x = x[:, :, 0] / x[:, :, 0].max()
        # plt.imshow(x)
        # plt.show()
        ref_mlem = reconstruction.MLEM(
            y_pet, x_init_pet.shape[:2], mlem_iterations, projector_id
        )
        ref_mlem = ref_mlem / ref_mlem.max()
        mse_mlem = utils.nrmse(data[index, :, :, 0], ref_mlem)
        ssim_mlem = ssim(ref_mlem, data[index, :, :, 0], data_range=ref_mlem.max())

        metrics_dlr["mse"].append(utils.nrmse(data[index, :, :, 0], x))
        metrics_dlr["ssim"].append(ssim(x, data[index, :, :, 0], data_range=x.max()))

        metrics_mlem["mse"].append(mse_mlem)
        metrics_mlem["ssim"].append(ssim_mlem)

    metrics = {"mlem": metrics_mlem, "dlr": metrics_dlr}
    return metrics
