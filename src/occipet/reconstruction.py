"""
Defines basic functions for image reconstruction from sinogram.
"""

import numpy as np
from .utils import *
from scipy.sparse.linalg import cg, LinearOperator
from scipy.fft import ifft2
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

    _, norm = back_projection(np.ones(y.shape,), projector_id)
    nabla_u = gradient(image)
    penalty_term = rho * div_2d(list(nabla_u - z_k + (gamma_k/rho)) )
    denominator = norm + penalty_term

    _,ybar = forward_projection(image, projector_id)
    ratio = div_zer(y, ybar)
    _, update = back_projection(ratio, projector_id)
    update = div_zer(update, denominator)
    return image*update

# TODO: shape of the linear operator (work with vector ?)
def merhanian_mr_step(
        s: np.ndarray, rho: float, z_k: np.ndarray,
        gamma_k: np.ndarray) -> np.ndarray:

    b = ( ifft2(s) + div_2d(list(rho * z_k - gamma_k)) ).flatten()
    A = LinearOperator((b.shape[0], s.flatten().shape[0]), matvec=partial(A_matrix_from_flatten, s.shape, rho))

    return cg(A, b)[0].reshape(s.shape)


def merhanian_constraint_step(correcting_z, comodal_z, current_z, sigma,
                              lambd, rho):

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


def merhanian_joint_pet_mr(rho_u, rho_v, lambda_u, lambda_v, sigma,
                           pet_number_iterations, mr_number_iterations,
                           number_iterations,
                           pet_data, mr_data, pet_shape, mr_shape,
                           projector_id):


    # Initialization
    epsilon = 10**(-12)
    u = np.ones(pet_shape)
    v = np.ones(mr_shape)
    gamma_u = np.repeat(np.array([u], copy=True), 2, axis=0)
    gamma_v = np.repeat(np.array([v], copy=True), 2, axis=0)

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
        temp_v = np.array(v, copy=True)
        for _ in range(mr_number_iterations):
            temp_v = merhanian_mr_step(mr_data, rho_v, z_v, gamma_v)
        v = temp_v

        # Constraint variable update
        correcting_z_u = gradient(u) + gamma_u/rho_u
        correcting_z_v = gradient(v) + gamma_v/rho_v

        alpha_u = np.sqrt(generalized_l2_norm_squared(z_v))/np.sqrt((generalized_l2_norm_squared(z_u) + epsilon))
        alpha_v = np.sqrt(generalized_l2_norm_squared(z_u))/np.sqrt((generalized_l2_norm_squared(z_v) + epsilon))

        z_u = merhanian_constraint_step(correcting_z_u, alpha_u * z_v, z_u, sigma,
                                        lambda_u, rho_u)

        z_v = merhanian_constraint_step(correcting_z_v, alpha_v * z_u, z_v, sigma,
                                        lambda_v, rho_v)

        # Lagrange multiplier update
        gamma_u = gamma_u + rho_u * (gradient(u) - z_u)
        gamma_v = gamma_v + rho_v * (gradient(v) - z_v)

    return u, v
