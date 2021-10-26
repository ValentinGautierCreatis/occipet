"""
Defines basic functions for image reconstruction from sinogram.
"""

import astra
import numpy as np


def create_projector(shape: tuple[int, int],
                     angles: np.ndarray, gpu: int) -> int:
    """ Create a projector for the given geometry

    :param shape: shape of the images in 2D
    :type shape: tuple[int, int]
    :param angles: array of angles used for projection
    :type angles: np.ndarray
    :param gpu: id of used gpu.
    :type gpu: int
    :returns: the astra id of the created projector

    """
    vol_geom = astra.create_vol_geom(shape)
    proj_geom = astra.create_proj_geom("parallel", 1.0, max(shape), angles)

    if gpu is None:
        projector_id = astra.create_projector('line', proj_geom, vol_geom)
    else:
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)

    return projector_id


def div_zer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Performs element wise division between 2 arrays

    :param a: numerator array
    :type a: np.ndarray
    :param b: denomiator array
    :type b: np.ndarray
    :returns: elementwise a/b

    """
    assert a.shape == b.shape, "both matrix should be the same size"
    epsilon = 10**(-12)
    size = a.shape
    new = np.zeros(size)
    new = a/(b+epsilon)
    return new


def forward_projection(x: np.ndarray,
                       projector_id: int, gpu: int = None
                       ) -> tuple[int, np.ndarray]:
    """ Forward projection using astra

    :param x: data on which is applied the forward projection
    :type x: np.ndarray
    :param projector_id: id of the used projector
    :type projector_id: int
    :param gpu: id of the used gpu
    :type gpu: int
    :returns: the forward projection of x

    """
    return astra.creators.create_sino(x, projector_id, gpuIndex=gpu)


def back_projection(y: np.ndarray, projector_id: int
                    ) -> tuple[int, np.ndarray]:
    """ Back projection using astra

    :param y: data on which is applied the retroprojection
    :type y: np.ndarray
    :param projector_id: id of the used projector
    :type projector_id: int
    :returns: the retroprojection of y

    """
    return astra.creators.create_backprojection(y, projector_id)


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
        _, ybar = forward_projection(image, projector_id)
        ratio = div_zer(y, ybar)
        _, update = back_projection(ratio, projector_id)
        update = div_zer(update, norm)
        image = image * update

    return image


def div_2d(q: list) -> np.ndarray:
    """ Computes the divergence of a 2D vector field

    :param q: 2D vector field represented as a list of two 2D matrices. (Ax(x,y), Ay(x,y))
    :type q: list
    :returns: The divergence of this vector field represented as a 2D ndarray.

    """
    grad1=np.gradient(q[0])
    grad2=np.gradient(q[1])
    return grad1[0]+grad2[1]


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
        _, ybar = forward_projection(image, projector_id)
        ratio = div_zer(y, ybar)
        _, update = back_projection(ratio, projector_id)
        update = div_zer(update, s)
        image = image * update

        tau = 1/(4*norm)

        g=[np.zeros(image_shape)]*2

        #computing lambda(n+1) with a weighted TV step
        for _ in range(175):
            #phi_int=[g[0],g[1]]
            div=div_2d(g)
            grad=np.gradient(wn*div-image)
            denom=1+tau*np.sqrt(grad[0]**2+grad[1]**2)
            g[0]=(g[0]+tau*grad[0])/denom
            g[1]=(g[1]+tau*grad[1])/denom

        image=image-wn*div_2d(g)
    return image
