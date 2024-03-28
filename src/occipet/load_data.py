"""
Module used to load data
"""

import nibabel as nib
import SimpleITK as sitk
import numpy as np
import brainweb
import pathlib
from pydicom import dcmread
from pydicom.fileset import FileSet
import pandas as pd
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



def generate_t1_mr_data_from_image(image: np.ndarray, noise_ratio: float
                              ):
    """generates noisy mri data from brainweb data file

    """
    signal = fft2(image)
    sigma = noise_ratio * np.amax(abs(signal))
    noise = np.random.normal(0, sigma, signal.shape)
    noisy_signal = signal + noise/10

    return noisy_signal, sigma


def generate_pet_data(data_file: str, background_event_ratio: float,
                      nb_angles: int = 100, nb_photons = 10**7):

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


def generate_pet_data_from_image(image: np.ndarray, background_event_ratio: float,
                      nb_angles: int = 100, nb_photons = 10**7) -> tuple[np.ndarray, int]:

    """generates noisy pet data from an image


    """
    pet = image * (nb_photons/(np.sum(image) * nb_angles))

    angles = np.linspace(0, 2*np.pi, nb_angles)
    projector_id = create_projector(pet.shape, angles, None)
    _, proj = forward_projection(pet, projector_id)
    proj = proj * (nb_photons/(np.sum(proj)))
    r = (1/(1/background_event_ratio - 1)) * np.ones(proj.shape) * np.sum(proj) / np.sum(np.ones(proj.shape))

    proj = proj + r

    return np.random.poisson(proj), projector_id


def get_image_from_dicom(path: str) -> np.ndarray:

    """Get image from dicom file

    :param path: path of the file
    :type path: str
    :returns: the read image

    """
    data = dcmread(path)
    return data.pixel_array


def make_dataset(fs: FileSet, list_keys: list, include_path=True
                 ) -> dict:
    """Generates a dataset from a list of keys in a dicomdir file set

    Parameters
    ----------
    fs : FileSet
        The FileSet from which to generate the dataset
    list_keys : list
        List of keys to extract from the file set
    include_path : Complete
        If True, adds an entry with the path to each file in the dataset

    Returns
    -------
    dict
        Dictionnary with keys from the input list of keys
        (+ path if include_path=True)

    """
    dic = {k: [] for k in list_keys}
    if include_path:
        dic["path"] = []
    for instance in fs:
        data = instance.load()
        for key in list_keys:
            dic[key].append(data[key].value)
        if include_path:
            dic["path"].append(instance.path)

    return dic


def get_matching_pairs(dicomdir_path: str) -> pd.DataFrame:
    """Generates a dataframe with the paths to each PET file and
    its corresponding MR file

    Parameters
    ----------
    dicomdir_path : str
        path to the DICOMDIR file

    Returns
    -------
    pd.DataFrame
        A DataFrame with 2 entries: "path" and "paired_mr" corresponding
        to the path to the PET file and its matching MR file
        respectively

    """
    fs = FileSet(dicomdir_path)
    keys = ["Modality", "SliceLocation"]
    df = pd.DataFrame.from_dict(make_dataset(fs, keys))
    pet = df[df["Modality"]=="PT"].sort_values("SliceLocation")
    mr = df[df["Modality"]=="MR"]
    pet["paired_mr"] = ""
    for ind, row in pet.iterrows():
        distances = mr["SliceLocation"].apply(lambda x: abs(x - row["SliceLocation"]))
        pet["paired_mr"][ind] = mr["path"][distances.idxmin()]
    return pet[["path", "paired_mr"]]


def make_image_set(pairs_df: pd.DataFrame) -> np.ndarray:
    """Creates a numpy array with PET/MR concatenated
    along the last dimension

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Dataframe containing a list of PET path with their corresponding
    MR image path

    Returns
    -------
    np.ndarray
        Complete

    """
    images = []
    for _, row in pairs_df.iterrows():
        pet = dcmread(row["path"]).pixel_array
        pet = pet.reshape(pet.shape + (1,))
        mr = dcmread(row["paired_mr"]).pixel_array
        mr = mr.reshape(mr.shape + (1,))
        multi_modal_image = np.concatenate((pet, mr) , axis=2)
        images.append(multi_modal_image)
    return np.array(images)


def create_patient_dicomdir(path: str) -> None:
    """Creates the DICOMDIR file for a given patient

    Parameters
    ----------
    path : str
        path to the patient folder with pet and mr sub folders

    """
    if (pathlib.Path(path) / "DICOMDIR").exists():
        return
    p = pathlib.Path(path).rglob("*.dcm")
    files = [x for x in p if x.is_file()]
    fs = FileSet()
    for f in files:
        fs.add(str(f))

    fs.write(str(path))


def resample(image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    out = sitk.Resample(image,
                         reference_image,
                         defaultPixelValue = 0.0)
    return out


def get_image(path: str):  # assumes that you want the series of the alphatically first entry in path

    file_name = str(list(pathlib.Path(path).iterdir())[0])
    data_directory = path

    # The block below matches all images in path with the same Series Instance UID as the first entry
    # Alternatively, this can be replaced by any other method producing a list of file names
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(str(file_name))
    file_reader.ReadImageInformation()
    series_ID = file_reader.GetMetaData('0020|000e')  # This gets the DICOM tag 0x0020000E (Series Instance UID)
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_ID)

    img = sitk.ReadImage(sorted_file_names)
    return img


def read_patient(patient_path: str):
    '''Read out pet and t1 from patient_path, resample one to the other

    :param patient_path: str: patient folder name
    :return: (Pet np.ndarray, T1 np.ndarray, resampled image np.ndarray
    '''
    img_pet = get_image(f'{patient_path}/pet')
    img_tone = get_image(f'{patient_path}/T1')

    resam = resample(img_tone, img_pet)  # The first image gets resampled, the second one provides the reference spacing, orientation, etc.

    # return images as numpy arrays, this can be replaced if you want to work in .nii, .mha, or other formats
    # in that case, use sitk.WriteImage(image: sitk.Image, filename: str)
    resampled_arr = sitk.GetArrayFromImage(resam)
    img_pet_arr = sitk.GetArrayFromImage(img_pet)
    img_tone_arr = sitk.GetArrayFromImage(img_tone)
    return img_pet_arr, img_tone_arr, resampled_arr
