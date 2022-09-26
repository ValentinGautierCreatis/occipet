"""
Module enabling scripts to be run in the command line. Functions defined
here are Typer wrapper around scripts from the file "scripts.py"
"""

from typer import Typer, Option
from .scripts import train_beta_vae_param, train_bimodal_vae_param, train_vae_param
from tools.parameters import Parameters

app = Typer()


@app.command("train_vae")
def train_vae_wrapper(params_path: str) -> None:
    parameters = Parameters.from_json(params_path)
    train_vae_param(parameters)


@app.command("train_beta_vae")
def train_beta_vae_wrapper(params_path: str) -> None:
    parameters = Parameters.from_json(params_path)
    train_beta_vae_param(parameters)


@app.command("train_bimodal_vae")
def train_bimodal_vae_wrapper(params_path: str) -> None:
    parameters = Parameters.from_json(params_path)
    train_bimodal_vae_param(parameters)
