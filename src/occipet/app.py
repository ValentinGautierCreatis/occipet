"""
Module enabling scripts to be run in the command line. Functions defined
here are Typer wrapper around scripts from the file "scripts.py"
"""

from typer import Typer
from .scripts import show_slice, generate_pet_phantom

app = Typer()


@app.command("show_slice")
def show_slice_wrapper(path: str, slice_id: int) -> None:
    """ Take an mnc data as input and display the wanted slice.

    :param path: Path to input mnc file
    :type path: str
    :param slice_id: Id of the slice we want to display
    :type slice_id: int
    :returns: None

    """
    show_slice(path, slice_id)


@app.command("generate_pet_phantom")
def generate_pet_phantom_wrapper(anatomical: str, slice_id: int):
    """ Generate a PET phantom from an anatomical mnc image

    :param anatomical: Path to the original anatomical image
    :type anatomical: str
    :param slice_id: Id of the slice we want to display
    :type slice_id: int
    :returns: None

    """
    generate_pet_phantom(anatomical, slice_id)
