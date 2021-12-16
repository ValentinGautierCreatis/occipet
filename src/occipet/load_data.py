"""
Module used to load data
"""

import nibabel as nib
import numpy as np
import brainweb
from scipy.fft import fft2, fftshift
from .utils import create_projector, forward_projection

def load_mnc(path: str) -> np.ndarray:
    """ Load data from a mnc file

    :param path: Path to the mnc file to be loaded
    :type path: str
    :returns: The 3D volume from the mnc file

    """
    img = nib.load(path)
    return img.get_fdata()


def generate_t1_mr_data(data_file, noise_ratio, noise_sigma):

    slices = (80,slice(120,230),slice(120,230))
    raw_data = brainweb.load_file(data_file)
    _,_,t1,_ = brainweb.toPetMmr(raw_data, "mMr", brainweb.FDG)
    t1 = t1[slices]
    t1_noisy = brainweb.noise(t1, noise_ratio, noise_sigma)
    transformed_t1 = fftshift(fft2(t1_noisy))

    return transformed_t1, t1, t1_noisy


def generate_pet_data(data_file, noise_ratio, noise_sigma):

    slices = (80,slice(120,230),slice(120,230))
    raw_data = brainweb.load_file(data_file)
    pet, *_ = brainweb.toPetMmr(raw_data, "mMr", brainweb.FDG)
    pet = pet[slices]

    angles = np.arange(0, 2*np.pi, 0.05)
    projector_id = create_projector(pet.shape, angles, None)
    _, proj = forward_projection(pet, projector_id)

    return np.random.poisson(proj), pet, projector_id
