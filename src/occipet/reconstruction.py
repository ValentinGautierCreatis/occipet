"""
Defines basic functions for image reconstruction from sinogram.
"""

import astra
import numpy as np
from .utils import *


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
