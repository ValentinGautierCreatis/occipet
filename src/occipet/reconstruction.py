"""
Defines basic functions for image reconstruction from sinogram.
"""

import numpy as np
from .utils import *
from scipy.sparse.linalg import cg, LinearOperator
from scipy.fft import ifft2, fft2
from functools import partial


def em_step(y: np.ndarray, image: np.ndarray,
            projector_id: int, norm: np.ndarray) -> np.ndarray:

    """Implements one step of expectation maximization

    :param y: Sinogram to reconstruct
    :type y: np.ndarray
    :param image: The current estimate of the image to reconstruct
    :type image: np.ndarray
    :param projector_id: id of the projector for this sinogram
    :type projector_id: int
    :param norm: Normalization factor for the geometry of this sinogram
    :type norm: np.ndarray
    :returns: Updated estimate of the image to reconstruct

    """
    _, ybar = forward_projection(image, projector_id)
    ratio = div_zer(y, ybar)
    _, update = back_projection(ratio, projector_id)
    update = div_zer(update, norm)
    return image * update


def merhanian_pet_step(y: np.ndarray, image: np.ndarray,
                       rho: float, z_k: np.ndarray,
                       gamma_k: np.ndarray, projector_id: int) -> np.ndarray:

    """One step of PET update for Merhanian algorithm as
    in equation (20)

    :param y: pet projections
    :type y: np.ndarray
    :param image: current image
    :type image: np.ndarray
    :param rho: parameter rho_u
    :type rho: float
    :param z_k: current z_u
    :type z_k: np.ndarray
    :param gamma_k: current gamma_u
    :type gamma_k: np.ndarray
    :param projector_id: id of the projector to be used
    :type projector_id: int
    :returns: The updated pet image

    """

    _, norm = back_projection(np.ones(y.shape,), projector_id)
    nabla_u = gradient(image)
    penalty_term = rho * div_2d(list(nabla_u - z_k + (gamma_k/rho)) )
    denominator = norm - penalty_term

    _,ybar = forward_projection(image, projector_id)
    ratio = div_zer(y, ybar)
    _, update = back_projection(ratio, projector_id)
    update = div_zer(update, denominator)
    return image*update


def merhanian_mr_step(
        s: np.ndarray, rho: float, z_k: np.ndarray,
        gamma_k: np.ndarray, nb_iterations: int,
        W: float) -> np.ndarray:

    """One step of MRI update for Merhanian algorithm using
    conjugate gradient as in equation (21)

    :param s: mri k-space data
    :type s: np.ndarray
    :param rho: parameter rho_v
    :type rho: float
    :param z_k: current z_v
    :type z_k: np.ndarray
    :param gamma_k: current gamma_v
    :type gamma_k: np.ndarray
    :param nb_iterations: maximum number of iterations
        for the conjugate gradient algorithm
    :type nb_iterations: int
    :param W: noise parameter
    :type W: np.ndarray
    :returns: The new MRI image

    """
    b = ( ifft2(W * s) - div_2d(list(rho * z_k - gamma_k)) ).flatten()
    A = LinearOperator((b.shape[0], s.flatten().shape[0]), matvec=partial(A_matrix_from_flatten, s.shape, rho, W))

    return cg(A, b, maxiter=nb_iterations)[0].reshape(s.shape)


def merhanian_constraint_step(correcting_z: np.ndarray,
                              comodal_z: np.ndarray, current_z: np.ndarray,
                              sigma: float, lambd: float, rho: float) -> np.ndarray:

    """On step of constraint variable z update for Merhanian
    algorithm as in equation (25) and (26). Works for
    both modalities as long as correct parameters are
    given.

    :param correcting_z: temporary variable defined in equation (24)
    :type correcting_z: np.ndarray
    :param comodal_z: rescaled z from the other modality
    :type comodal_z: np.ndarray
    :param current_z: current z of the current modality
    :type current_z: np.ndarray
    :param sigma: parameter of the regularization function as
        defined in equation (8)
    :type sigma: float
    :param lambd: regularization parameter
    :type lambd: float
    :param rho: parameter rho of the current modality
    :type rho: float
    :returns: The updated constraint variable

    """
    norm_matrix = co_norm(correcting_z, comodal_z)
    omega = np.exp(-sigma * co_norm(current_z, comodal_z))
    update_matrix = div_zer((norm_matrix - (lambd/rho)*omega).clip(min=0), norm_matrix)
    return multiply_along_0axis(update_matrix, correcting_z)


def MLEM(y: np.ndarray, image_shape: np.ndarray,
         nb_iterations: int, projector_id: int) -> np.ndarray:
    """ MLEM implementation using astra

    :param y: sinogram to reconstruct
    :type y: np.ndarray
    :param image_shape: shape of the image to be reconstructed
    :type image_shape: np.ndarray
    :param nb_iterations: number of iterations for the algorithm
    :type nb_iterations: int
    :param projector_id: id of the projector for this sinogram
    :type projector_id: int
    :returns: the reconstructed image

    """
    image = np.ones(image_shape)
    _, norm = back_projection(np.ones(y.shape), projector_id)
    assert norm.shape == image_shape, \
        "The image shape is different from the one specified by the projector"

    for _ in range(nb_iterations):
        image = em_step(y, image, projector_id, norm)
    return image[1:-1,:]


def EMTV(y: np.ndarray, image_shape: np.ndarray,
         nb_iterations: int, projector_id: int,
         alpha: float=0.75) -> np.ndarray:

    """ TV regularized MLEM from Swatzky et al, Accurate EM-TV algorithm in
     PET with low SNR

    :param y: sinogram to reconstruct
    :type y: np.ndarray
    :param image_shape: shape of the image to be reconstructed
    :type image_shape: np.ndarray
    :param nb_iterations: number of titerations for the algorithm
    :type nb_iterations: int
    :param projector_id: id of the projector for this sinogram
    :type projector_id: int
    :param alpha: weight of the TV regularization
    :type alpha: float
    :returns: the reconstructed image

    """

    image = np.ones(image_shape)
    _, s = back_projection(np.ones(y.shape), projector_id)
    assert s.shape == image_shape, \
        "The image shape is different from the one specified by the projector"

    for _ in range(nb_iterations):
        # wn corresponds to alpha_tilde * uk in the paper
        wn=alpha*div_zer(image, s)
        norm=np.max(wn)

        #computing lambda(n+1/2) with a EM step
        image = em_step(y, image, projector_id, s)

        tau = 1/(4*norm)

        g=[np.zeros(image_shape)]*2

        #computing lambda(n+1) with a weighted TV step
        for _ in range(175):
            #phi_int=[g[0],g[1]]
            div=div_2d(g)
            grad=gradient(wn*div-image)
            denom=1+tau*np.sqrt(grad[0]**2+grad[1]**2)
            g[0]=(g[0]+tau*grad[0])/denom
            g[1]=(g[1]+tau*grad[1])/denom

        image=image-wn*div_2d(g)
    return image[1:-1, :]


def merhanian_joint_pet_mr(rho_u: float, rho_v: float, lambda_u: float, lambda_v: float,
                           sigma: float,
                           pet_number_iterations: int, mr_number_iterations: int,
                           number_iterations: int,
                           pet_data: np.ndarray, mr_data: np.ndarray,
                           pet_shape: np.ndarray, mr_shape: np.ndarray,
                           projector_id: int, W: float
                           ) -> tuple[np.ndarray, np.ndarray]:

    """Joint reconstruction for PET and MRI data based on
    Merhanian algorithm: https://pubmed.ncbi.nlm.nih.gov/28436851/

    :param rho_u: parameter rho_u
    :type rho_u: float
    :param rho_v: parameter rho_v
    :type rho_v: float
    :param lambda_u: parameter lambda_u
    :type lambda_u: float
    :param lambda_v: parameter lambda_v
    :type lambda_v: float
    :param sigma: parameter of the regularization function as
        defined in equation (8)
    :type sigma: float
    :param pet_number_iterations: number of iterations for the
        PET update
    :type pet_number_iterations: int
    :param mr_number_iterations: number of iterations for the
        conjugate gradient algorithm for the MRI update
    :type mr_number_iterations: int
    :param number_iterations: number of iterations for the algorithm
    :type number_iterations: int
    :param pet_data: pet projections
    :type pet_data: np.ndarray
    :param mr_data: mri k-space data
    :type mr_data: np.ndarray
    :param pet_shape: shape of the resulting pet image
    :type pet_shape: np.ndarray
    :param mr_shape: shape of the resulting mri image
    :type mr_shape: np.ndarray
    :param projector_id: id of the projector to be used
    :type projector_id: int
    :param W: noise
    :type W: float
    :returns: The reconstructed pet and mri images

    """


    # Initialization
    epsilon = 10**(-12)
    u = np.ones(pet_shape)
    v = np.zeros(mr_shape)
    gamma_u = np.ones((len(pet_shape),) + pet_shape)
    gamma_v = np.ones((len(mr_shape),) + mr_shape)

    for _ in range(number_iterations):

        z_u = gradient(u)
        z_v = gradient(v)

        # pet update
        temp_u = np.array(u, copy=True)
        for _ in range(pet_number_iterations):
            temp_u = merhanian_pet_step(pet_data, temp_u, rho_u, z_u, gamma_u,
                                        projector_id)
        u = temp_u

        # mr update
        v = merhanian_mr_step(mr_data, rho_v, z_v, gamma_v, mr_number_iterations, W)

        # Constraint variable update
        correcting_z_u = gradient(u) + gamma_u/rho_u
        correcting_z_v = gradient(v) + gamma_v/rho_v

        alpha_u = np.sqrt(generalized_l2_norm_squared(z_v))/(np.sqrt(generalized_l2_norm_squared(z_u)) + epsilon)
        alpha_v = np.sqrt(generalized_l2_norm_squared(z_u))/(np.sqrt(generalized_l2_norm_squared(z_v)) + epsilon)

        temp_z_u = merhanian_constraint_step(correcting_z_u, alpha_v * z_v, z_u, sigma,
                                        lambda_u, rho_u)

        z_v = merhanian_constraint_step(correcting_z_v, alpha_u * z_u, z_v, sigma,
                                        lambda_v, rho_v)

        z_u = temp_z_u

        # Lagrange multiplier update
        gamma_u = gamma_u + rho_u * (gradient(u) - z_u)
        gamma_v = gamma_v + rho_v * (gradient(v) - z_v)

    return  u, v


def courbes_joint_pet_mr(rho_u, rho_v, lambda_u, lambda_v, sigma,
                           pet_number_iterations, mr_number_iterations,
                           number_iterations,
                           pet_data, mr_data, pet_shape, mr_shape,
                           projector_id,
                           pet, mri, W):


    # Initialization
    list_keys = ["reconstruction_error", "joint_tv", "data_fidelity", "alpha", "froebenius"]
    points_pet = dict((k, list()) for k in list_keys)
    points_mri = dict((k, list()) for k in list_keys)

    epsilon = 10**(-12)
    u = np.ones(pet_shape)
    v = np.zeros(mr_shape)
    gamma_u = np.ones((len(pet_shape),) + pet_shape)
    gamma_v = np.ones((len(mr_shape),) + mr_shape)

    for i in range(number_iterations):

        points_pet["data_fidelity"].append(data_fidelity_pet(u, pet_data, projector_id))
        points_mri["data_fidelity"].append(data_fidelity_mri(v, mr_data, W))

        z_u = gradient(u)
        z_v = gradient(v)

        # pet update
        temp_u = np.array(u, copy=True)
        for _ in range(pet_number_iterations):
            temp_u = merhanian_pet_step(pet_data, temp_u, rho_u, z_u, gamma_u,
                                        projector_id)
        u = temp_u

        # mr update
        v = merhanian_mr_step(mr_data, rho_v, z_v, gamma_v, mr_number_iterations, W)

        # Constraint variable update
        correcting_z_u = gradient(u) + gamma_u/rho_u
        correcting_z_v = gradient(v) + gamma_v/rho_v

        alpha_u = np.sqrt(generalized_l2_norm_squared(z_v))/(np.sqrt(generalized_l2_norm_squared(z_u)) + epsilon)
        alpha_v = np.sqrt(generalized_l2_norm_squared(z_u))/(np.sqrt(generalized_l2_norm_squared(z_v)) + epsilon)

        temp_z_u = merhanian_constraint_step(correcting_z_u, alpha_v * z_v, z_u, sigma,
                                        lambda_u, rho_u)

        z_v = merhanian_constraint_step(correcting_z_v, alpha_u * z_u, z_v, sigma,
                                        lambda_v, rho_v)

        z_u = temp_z_u

        # Lagrange multiplier update
        gamma_u = gamma_u + rho_u * (gradient(u) - z_u)
        gamma_v = gamma_v + rho_v * (gradient(v) - z_v)


        points_pet["reconstruction_error"].append(np.sum(abs(u-pet)))
        points_pet["joint_tv"].append(np.sum(co_norm(z_u, alpha_v * z_v)))
        points_pet["alpha"].append(alpha_u)
        points_pet["froebenius"].append(generalized_l2_norm_squared(z_u))

        points_mri["reconstruction_error"].append(np.sum(abs(v-mri)))
        points_mri["joint_tv"].append(np.sum(co_norm(alpha_u * z_u, z_v)))
        points_mri["alpha"].append(alpha_v)
        points_mri["froebenius"].append(generalized_l2_norm_squared(z_v))

    return points_pet, points_mri, u, v
