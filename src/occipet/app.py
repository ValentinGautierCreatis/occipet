from typer import Typer
from .scripts import show_slice, generate_pet_phantom

app = Typer()

@app.command("show_slice")
def show_slice_wrapper(path: str, slice_id: int) -> None:
    show_slice(path, slice_id)

@app.command("generate_pet_phantom")
def generate_pet_phantom_wrapper(anatomical: str, slice_id: int):
    generate_pet_phantom(anatomical, slice_id)
