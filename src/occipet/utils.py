#!/usr/bin/env python3

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
    # TODO check this max(shape) and see if we can get the shape from this
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


def gradient(a) -> np.ndarray :
    '''
    Compute gradient of a

    Parameters
    ----------
    a : cp.array
        data

    Returns
    -------
    grad : cp.array
           gradient(a)
    '''
    dim = len(a.shape)
    if dim == 2 :
        grad = np.zeros((dim,a.shape[0],a.shape[1]))
        grad[0] = np.diff(a,1,axis=0,append=a[-1,:].reshape(1,a.shape[1]))
        grad[1] = np.diff(a,1,axis=1,append=a[:,-1].reshape(a.shape[0],1))
    elif dim == 3 :
        grad = np.zeros((3,a.shape[0],a.shape[1],a.shape[2]))
        grad[0] = np.diff(a,1,axis=0,append=a[-1,:,:].reshape(1,a.shape[1],a.shape[2]))
        grad[1] = np.diff(a,1,axis=1,append=a[:,-1,:].reshape(a.shape[0],1,a.shape[2]))
        grad[2] = np.diff(a,1,axis=2,append=a[:,:,-1].reshape(a.shape[0],a.shape[1],1))
    else:
        raise IndexError("Unvalid dimension for a. Must be 2 or 3")
    return grad


def gradient_div(a) :
    '''
    Compute gradient of a used in divergence

    Parameters
    ----------
    a : cp.array
        data

    Returns
    -------
    grad : cp.array
           gradient(a)
    '''
    dim = len(a.shape)
    if dim == 2 :
        grad = np.zeros((2,a.shape[0],a.shape[1]))
        grad[0] = np.diff(a,1,axis=0,prepend=a[0,:].reshape(1,a.shape[1]))
        grad[1] = np.diff(a,1,axis=1,prepend=a[:,0].reshape(a.shape[0],1))
    elif dim == 3 :
        grad = np.zeros((3,a.shape[0],a.shape[1],a.shape[2]))
        grad[0] = np.diff(a,1,axis=0,prepend=a[0,:,:].reshape(1,a.shape[1],a.shape[2]))
        grad[1] = np.diff(a,1,axis=1,prepend=a[:,0,:].reshape(a.shape[0],1,a.shape[2]))
        grad[2] = np.diff(a,1,axis=2,prepend=a[:,:,0].reshape(a.shape[0],a.shape[1],1))
    else:
        raise IndexError("Unvalid dimension for a. Must be 2 or 3")
    return grad


def div_2d(q: list) -> np.ndarray:
    """ Computes the divergence of a 2D vector field

    :param q: 2D vector field represented as a list of two 2D matrices. (Ax(x,y), Ay(x,y))
    :type q: list
    :returns: The divergence of this vector field represented as a 2D ndarray.

    """
    grad1=gradient_div(q[0])
    grad2=gradient_div(q[1])
    return grad1[0]+grad2[1]


def merhanian_A_matrix(rho, image):
    return rho * div_2d(list(gradient(image)))


def generalized_l2_norm_squared(vector):
    return np.sum(vector**2)
