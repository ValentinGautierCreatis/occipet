"""
Module enabling scripts to be run in the command line. Functions defined
here are Typer wrapper around scripts from the file "scripts.py"
"""

from typer import Typer, Option
from .scripts import train_vae

app = Typer()


@app.command("train_vae")
def train_vae_wrapper(checkpoint_dir: str,
                      data_path: str) -> None:

    train_vae(checkpoint_dir, data_path)
