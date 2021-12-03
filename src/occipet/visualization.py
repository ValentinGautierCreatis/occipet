import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def show_array(data: np.ndarray, slice_id: int, axis: int = 0) -> None:
    """Displays slices of a 3D colume, allowing to move along selected axis

    :param data: 3D array to be displayed
    :type data: np.ndarray
    :param slice_id: Slice of the array to be displayed first
    :type slice_id: int
    :param axis: axis along which we want to be able to move
    :type axis: int
    :returns: None

    """
    slices = [ slice(0,data.shape[i]) for i in range(len(data.shape)) ]
    slices[axis] = slice_id

    # Defining the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    image = ax.imshow(data[tuple(slices)], cmap="gray")


    # Defining the slider
    ax_color = 'lightgoldenrodyellow'
    ax_slice = plt.axes([0.25,0.1,0.65,0.03], facecolor=ax_color)

    slider_slice = Slider(ax_slice, 'ID slice', 0, data.shape[axis] - 1, valinit=slice_id, valstep=1)

    # Update function for the plot
    def update(val):
        slice_id = slider_slice.val
        slices[axis] = slice_id
        image.set_data(data[tuple(slices)])
        fig.canvas.draw_idle()

    # Ploting
    slider_slice.on_changed(update)
    plt.show()
