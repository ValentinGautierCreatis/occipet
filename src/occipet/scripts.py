"""
Collection of scripts that can be used to automate tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
from .reconstruction import MLEM, forward_projection, create_projector, EMTV
from .load_data import load_mnc
from matplotlib.widgets import Slider


def show_slice(path: str, slice_id: int) -> None:
    """ Take an mnc data as input and display the wanted slice.

    :param path: Path to input mnc file
    :type path: str
    :param slice_id: Id of the slice we want to display
    :type slice_id: int
    :returns: None

    """
    # Loading data and checking
    data = load_mnc(path)
    assert 0 <= slice_id < data.shape[0], \
        "f{slice_id} is not a valid slice ID"

    # Defining the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    image = ax.imshow(data[slice_id, :, :], cmap="gray")


    # Defining the slider
    ax_color = 'lightgoldenrodyellow'
    ax_slice = plt.axes([0.25,0.1,0.65,0.03], facecolor=ax_color)

    slider_slice = Slider(ax_slice, 'ID slice', 0, data.shape[0] - 1, valinit=slice_id, valstep=1)

    # Update function for the plot
    def update(val):
        slice_id = slider_slice.val
        image.set_data(data[slice_id, :, :])
        fig.canvas.draw_idle()

    # Ploting
    slider_slice.on_changed(update)
    plt.show()


def generate_pet_phantom(anatomical: str, slice_id: int) -> None:

    """ Generate a PET phantom from an anatomical mnc image

    :param anatomical: Path to the original anatomical image
    :type anatomical: str
    :param slice_id: Id of the slice we want to display
    :type slice_id: int
    :returns: None

    """

    # Parameters
    angles = np.arange(0, 2*np.pi, 0.05)
    nb_iterations = 200
    bkg_event_ratio = 0.2

    # Load data
    data = load_mnc(anatomical)
    assert 0 < slice_id < data.shape[0], f"{slice_id} is not a valid slice id"
    image = data[slice_id, :, :]

    # Making PET phantom
    pet_phantom = np.zeros(image.shape)
    pet_phantom[image == 2] = 4
    pet_phantom[image == 3] = 1
    pet_phantom[image == 4] = 0
    pet_phantom[image == 5] = 0
    pet_phantom[image == 6] = 0
    pet_phantom[image == 8] = 1/3
    pet_phantom[image == 9] = 0

    # Generate PET scanner data
    projector_id = create_projector(pet_phantom.shape, angles, None)

    _, proj = forward_projection(pet_phantom, projector_id)
    r = (1/(1/bkg_event_ratio - 1))*np.ones(proj.shape)\
        * np.sum(proj)/np.sum(np.ones(proj.shape))

    y_nonoise = proj + r
    y = np.random.poisson(y_nonoise)

    # PET reconstruction

    x_recon = EMTV(y, pet_phantom.shape, nb_iterations, projector_id)

    plt.imshow(x_recon, cmap="gray")
    plt.show()
