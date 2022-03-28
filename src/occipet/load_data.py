"""
Module used to load data
"""

import nibabel as nib
import numpy as np
import brainweb
import pydicom
from scipy.fft import fft2
from .utils import create_projector, div_zer, forward_projection, back_projection

def load_mnc(path: str) -> np.ndarray:
    """ Load data from a mnc file

    :param path: Path to the mnc file to be loaded
    :type path: str
    :returns: The 3D volume from the mnc file

    """
    img = nib.load(path)
    return img.get_fdata()


def generate_t1_mr_data(data_file, noise_ratio):
    """
    deprecated
    """

    slices = (80,slice(120,230),slice(120,230))
    raw_data = brainweb.load_file(data_file)
    _,_,t1,_ = brainweb.toPetMmr(raw_data, "mMr", brainweb.FDG)
    t1 = t1[slices]
    transformed_t1 = fft2(t1)

    signal_power = np.sum(abs(transformed_t1)**2)/transformed_t1.size
    signal_power_db = 10*np.log10(signal_power)

    noise_power_db = signal_power_db - noise_ratio
    noise_power = 10**(noise_power_db/10)

    noise_real = np.random.normal(0, 1, transformed_t1.shape)*np.sqrt(noise_power/2)
    noise_imaginary = np.random.normal(0, 1, transformed_t1.shape)*np.sqrt(noise_power/2)

    noise = noise_real + 1j*noise_imaginary

    noisy_signal = transformed_t1 + noise

    return t1, noisy_signal

def generate_t1_mr_data_sigma(data_file: str, noise_ratio: float
                              ):
    """generates noisy mri data from brainweb data file

    :param data_file: brainweb datafile
    :type data_file: np.ndarray
    :param noise_ratio: ratio of noise on the projection data
    :type noise_ratio: float
    :returns: the original image, the noisy_projections and the added noise

    """
    slices = (80,slice(120,230),slice(120,230))
    raw_data = brainweb.load_file(data_file)
    _,_,t1,_ = brainweb.toPetMmr(raw_data, "mMr", brainweb.FDG)
    t1 = t1[slices]

    signal = fft2(t1)
    sigma = noise_ratio * np.amax(abs(signal))
    noise = np.random.normal(0, sigma, signal.shape)
    noisy_signal = signal + noise/10

    return t1, noisy_signal, sigma


def generate_pet_data(data_file: str, background_event_ratio: float,
                      nb_angles: int = 100, nb_photons = 1000):

    """generates noisy pet data from brainweb data file

    :param data_file: brainweb datafile
    :type data_file: str
    :param background_event_ratio: ratio of background event noise to add
    :type background_event_ratio: float
    :returns:  the original image, the noisy data and the projector for these data

    """
    slices = (80,slice(120,230),slice(120,230))
    raw_data = brainweb.load_file(data_file)
    pet, *_ = brainweb.toPetMmr(raw_data, "mMr", brainweb.FDG)
    pet = pet[slices]
    pet = pet * (nb_photons/(np.sum(pet) * nb_angles))

    angles = np.linspace(0, 2*np.pi, nb_angles)
    projector_id = create_projector(pet.shape, angles, None)
    _, proj = forward_projection(pet, projector_id)
    r = (1/(1/background_event_ratio - 1)) * np.ones(proj.shape) * np.sum(proj) / np.sum(np.ones(proj.shape))

    proj = proj + r

    return pet, np.random.poisson(proj), projector_id


def get_image_from_dicom(path: str) -> np.ndarray:

    """Get image from dicom file

    :param path: path of the file
    :type path: str
    :returns: the read image

    """
    data = pydicom.dcmread(path)
    return data.pixel_array

