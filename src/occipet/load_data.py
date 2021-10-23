"""
Module used to load data
"""

import nibabel as nib
import numpy as np


def load_mnc(path: str) -> np.ndarray:
    """ Load data from a mnc file

    :param path: Path to the mnc file to be loaded
    :type path: str
    :returns: The 3D volume from the mnc file

    """
    img = nib.load(path)
    return img.get_fdata()
