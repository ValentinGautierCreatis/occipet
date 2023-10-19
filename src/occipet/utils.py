#!/usr/bin/env python3

import astra
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def create_projector(shape: tuple[int, int], angles: np.ndarray, gpu: int) -> int:
    """Create a projector for the given geometry

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
    proj_geom = astra.create_proj_geom("parallel", 1, max(shape), angles)

    if gpu is None:
        projector_id = astra.create_projector("line", proj_geom, vol_geom)
    else:
        projector_id = astra.create_projector("cuda", proj_geom, vol_geom)

    return projector_id


def div_zer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs element wise division between 2 arrays

    :param a: numerator array
    :type a: np.ndarray
    :param b: denomiator array
    :type b: np.ndarray
    :returns: elementwise a/b

    """
    assert a.shape == b.shape, "both matrix should be the same size"
    epsilon = 10 ** (-12)
    size = a.shape
    new = np.zeros(size)
    non_zero = b != 0
    new[non_zero] = a[non_zero] / b[non_zero]
    # new = a/(b+epsilon)
    return new


def forward_projection(
    x: np.ndarray, projector_id: int, gpu: int = None
) -> tuple[int, np.ndarray]:
    """Forward projection using astra

    :param x: data on which is applied the forward projection
    :type x: np.ndarray
    :param projector_id: id of the used projector
    :type projector_id: int
    :param gpu: id of the used gpu
    :type gpu: int
    :returns: the forward projection of x

    """
    return astra.creators.create_sino(x, projector_id, gpuIndex=gpu)


def back_projection(y: np.ndarray, projector_id: int) -> tuple[int, np.ndarray]:
    """Back projection using astra

    :param y: data on which is applied the retroprojection
    :type y: np.ndarray
    :param projector_id: id of the used projector
    :type projector_id: int
    :returns: the retroprojection of y

    """
    return astra.creators.create_backprojection(y, projector_id)


def gradient(a) -> np.ndarray:
    """
    Compute gradient of a

    Parameters
    ----------
    a : cp.array
        data

    Returns
    -------
    grad : cp.array
           gradient(a)
    """
    dim = len(a.shape)
    if dim == 2:
        grad = np.zeros((dim, a.shape[0], a.shape[1]), dtype=a.dtype)
        grad[0] = np.diff(a, 1, axis=0, append=a[-1, :].reshape(1, a.shape[1]))
        grad[1] = np.diff(a, 1, axis=1, append=a[:, -1].reshape(a.shape[0], 1))
    elif dim == 3:
        grad = np.zeros((3, a.shape[0], a.shape[1], a.shape[2]))
        grad[0] = np.diff(
            a, 1, axis=0, append=a[-1, :, :].reshape(1, a.shape[1], a.shape[2])
        )
        grad[1] = np.diff(
            a, 1, axis=1, append=a[:, -1, :].reshape(a.shape[0], 1, a.shape[2])
        )
        grad[2] = np.diff(
            a, 1, axis=2, append=a[:, :, -1].reshape(a.shape[0], a.shape[1], 1)
        )
    else:
        raise IndexError(f"Unvalid dimension {dim} for a. Must be 2 or 3")
    return grad


def gradient_div(a):
    """
    Compute gradient of a used in divergence

    Parameters
    ----------
    a : cp.array
        data

    Returns
    -------
    grad : cp.array
           gradient(a)
    """
    dim = len(a.shape)
    if dim == 2:
        grad = np.zeros((2, a.shape[0], a.shape[1]), dtype=a.dtype)
        grad[0] = np.diff(a, 1, axis=0, prepend=a[0, :].reshape(1, a.shape[1]))
        grad[1] = np.diff(a, 1, axis=1, prepend=a[:, 0].reshape(a.shape[0], 1))
    elif dim == 3:
        grad = np.zeros((3, a.shape[0], a.shape[1], a.shape[2]))
        grad[0] = np.diff(
            a, 1, axis=0, prepend=a[0, :, :].reshape(1, a.shape[1], a.shape[2])
        )
        grad[1] = np.diff(
            a, 1, axis=1, prepend=a[:, 0, :].reshape(a.shape[0], 1, a.shape[2])
        )
        grad[2] = np.diff(
            a, 1, axis=2, prepend=a[:, :, 0].reshape(a.shape[0], a.shape[1], 1)
        )
    else:
        raise IndexError("Unvalid dimension for a. Must be 2 or 3")
    return grad


# def gradient(x):
#     return np.array(np.gradient(x))

# gradient_div = np.gradient


def div_2d(q: list) -> np.ndarray:
    """Computes the divergence of a 2D vector field

    :param q: 2D vector field represented as a list of two 2D matrices. (Ax(x,y), Ay(x,y))
    :type q: list
    :returns: The divergence of this vector field represented as a 2D ndarray.

    """
    grad1 = gradient_div(q[0])
    grad2 = gradient_div(q[1])
    return grad1[0] + grad2[1]


def merhanian_A_matrix(rho: float, W: float, image: np.ndarray) -> np.ndarray:
    """Computes de effect of the A matrix of the system in equation (21)
    on a given input vector

    :param rho: parameter rho_v
    :type rho: float
    :param W: noise
    :type W: float
    :param image: current image
    :type image: np.ndarray
    :returns: the image on which we have applied the A matrix

    """
    return ifft2(W * fft2(image)) - rho * div_2d(list(gradient(image)))


def A_matrix_from_flatten(
    shape_image: np.ndarray, rho: float, W: float, flat_image: np.ndarray
) -> np.ndarray:
    """Takes the input image as a vector and compute the effect of the A
    matrix of the system in equation (21)

    :param shape_image: shape of the image
    :type shape_image: np.ndarray
    :param rho: parameter rho_v
    :type rho: float
    :param W: noise
    :type W: float
    :param flat_image: the flattened image
    :type flat_image: np.ndarray
    :returns: the flattened image on which we have applied the A matrix

    """
    image = flat_image.reshape(shape_image)
    image = merhanian_A_matrix(rho, W, image)
    return image.flatten()


def generalized_l2_norm_squared(vector: np.ndarray) -> float:
    """Squared Froebenius norm of the input

    :param vector: input vector/matrix
    :type vector: np.ndarray
    :returns: the Froebenius norm

    """
    return np.sum(abs(vector) ** 2)


def co_norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Computes the co norm of two gradient like objects on each voxel/pixel
    equation (7) of Merhanian's paper

    :param u: gradient like object
    :type u: np.ndarray
    :param v: gradient like object
    :type v: np.ndarray
    :returns: A matrix representing the co norm computed on each voxel

    """
    u_norm_squared = np.sum(abs(u) ** 2, axis=0)
    v_norm_squared = np.sum(abs(v) ** 2, axis=0)
    return np.sqrt(u_norm_squared + v_norm_squared)


def multiply_along_0axis(multiplier: np.ndarray, multiplied: np.ndarray) -> np.ndarray:
    """Take a matrix and multiply each element of an array of matrix of same
    dimension by this matrix

    :param multiplier: the matrix to use as a multiplier
    :type multiplier: np.ndarray
    :param multiplied: array of matrix to be multiplied
    :type multiplied: np.ndarray
    :returns: np array of same dimensions as multiplied.

    """
    new = np.zeros(multiplied.shape, dtype=multiplied.dtype)
    for k in range(multiplied.shape[0]):
        new[k] = multiplier * multiplied[k]
    return new


def data_fidelity_pet(image: np.ndarray, data: np.ndarray, projector_id: int) -> float:
    """Compute the data fidelity term for pet data as in
    equation (5) of Merhanian's paper


    :param image: image we want to check the data fidelity of
    :type image: np.ndarray
    :param data: reference projections data
    :type data: np.ndarray
    :param projector_id: id of the projector to be used
    :type projector_id: int
    :returns: the data fidelity term

    """
    _, projected_image = forward_projection(image, projector_id)
    intermediate = projected_image - data * np.log(projected_image)
    return np.sum(intermediate)


def data_fidelity_mri(image: np.ndarray, data: np.ndarray, W: float) -> float:
    """Compute the data fidelity term for mri data as in
    equation (6) of Merhanian's paper

    :param image: image we want to check the data fidelity of
    :type image: np.ndarray
    :param data: reference projections data
    :type data: np.ndarray
    :param W: noise
    :type W: float
    :returns: the data fidelity term

    """
    projected_image = fft2(image)
    intermediate = W * abs(projected_image - data) ** 2
    return np.sum(intermediate)


def nrmse(actual, predicted):
    """Computes the NRMSE between actual and predicted

    Parameters
    ----------
    actual : np.ndarray
        The reference image
    predicted : np.ndarray
        The obtained image

    """
    assert actual.shape == predicted.shape, "Both images must have the same dimensions"

    rmse = np.sqrt(np.mean((predicted - actual)**2))

    # Calculate the range of the data
    data_range = np.max(actual) - np.min(actual)

    # Calculate NRMSE
    nrmse = rmse / data_range

    return nrmse


def add_gaussian_noise(image, psnr=20):
    image_power = np.sum(image**2) / image.size
    image_power_db = 10 * np.log10(image_power)

    noise_power_db = image_power_db - psnr
    noise_power = 10 ** (noise_power_db / 10)

    noise = np.random.normal(0, 1, image.shape) * np.sqrt(noise_power)

    return image + noise


def normalize_meanstd(a, axis=None) -> np.ndarray:
    """Returns the standardized version of the input array along desired axes

    Parameters
    ----------
    a : np.ndarray
        The array to be standardized
    axis : int, tuple
        Axes along which to perform reduction

    Returns
    -------
    np.ndarray
        Standardized array

    """
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean) ** 2).mean(axis=axis, keepdims=True))
    return (a - mean) / std


def normalize(a, axis=None) -> np.ndarray:
    """Returns the normalized version of the input array along desired axes

    Parameters
    ----------
    a : np.ndarray
        The array to be standardized
    axis : int, tuple
        Axes along which to perform reduction

    Returns
    -------
    np.ndarray
        Normalized array

    """
    mini = np.min(a, axis=axis, keepdims=True)
    maxi = np.max(a, axis=axis, keepdims=True)
    return (a - mini) / (maxi - mini)
