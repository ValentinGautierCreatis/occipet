# Occipet

Occipet is a python package regrouping the code I wrote during my PhD. It contains functions for PET and MRI reconstruction as well as other tools for vizualisation or handling data. Alongside Occipet is a deep_learning package containing the implementation of several deep learning models, mainly VAEs and diffusion models, some of them being used in the reconstruction process in Occipet. It also exposes several scripts that can be called from the command line thanks to [Typer](https://typer.tiangolo.com/).

This code is the one used in the following publications : 

* Valentin Gautier, Alexandre Bousse, Florent Sureau, Claude Comtat, Voichita Maxim, et al.. Bimodal PET/MRI generative reconstruction based on VAE architectures. Physics in Medicine and Biology, 2024, 69 (24), pp.245019. ⟨10.1088/1361-6560/ad9133⟩.

* Valentin Gautier, Claude Comtat, Florent Sureau, Alexandre Bousse, Voichita Maxim, et al.. Synergistic PET/MR reconstruction with VAE constraint. EUSIPCO 2024 - 32nd European Signal Processing Conference, Aug 2024, Lyon, France. 2024.

* Valentin Gautier, Claude Comtat, Florent Sureau, Alexandre Bousse, Louise Friot–giroux, et al.. VAE constrained MR guided PET reconstruction. 17th International Meeting on Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine (Fully3D), Jul 2023, Stony Brook (NY), United States. 2023.

## Overview of the code

### occipet

Occipet contains the code relative to the reconstruction of PET and MR images. 

* `load_data.py` contains a set of functions used for loading data and simulating PET or MRI data. 
* `polar.py` contains functions to perform radial subsampling used for the simulated MRI data.
* `visualization.py` is a tool for inspecting volumes.
* `reconstruction.py` contains functions for reconstructing PET and MRI images using different algorithms. 
* `deep_latent_reconstruction.py` contains the main contribution of this PhD and its variations. 
* `garage.py` contains several tests for debugging and better understanding DLR.
* `scripts.py` contains utility functions that a user might want to use outside of the code and `app.py` offers a command line interface to these functions that can be used once the library is installed.
* `utils.py` contains functions that are used across the different files.

### deep_learning 

This module is composed of many different deep learning models that are implemented as Classes. Most of them are VAEs and the submodule `diffusion` contains everything that is diffusion models related.

`variational_auto_encoder.py` is the first and the simplest implementation of the tested VAEs. It is improved in `vae1_2.py` which uses one decoder per modality. `adaptive1_2.py` implements a beta-VAE with an automatic beta selection method. The implementation of this method has not been verified yet so there is no guarantee of it working properly. `test_vae.py` contains some early VAE prototypes. The rest is composed of the other tested architectures. 

## Installation 

First, clone this repository:

``` sh
git clone git@github.com:ValentinGautierCreatis/occipet.git
```

You can install it in development mode (I recommend installing it in a virtual evironment using your favorite tool such as conda, venv, poetry, pyenv , ...). Changes in the source files will be applied without having to install the package again. To do so, run the following :

``` sh
pip install -e .
```

This does not install astra-toolbox which is the package used to manage the projections for the PET reconstruction. It seems the build available on PyPI is broken at the moment so you need to manually build it. Follow the instructions given on the readme of the package on their official github page: https://github.com/astra-toolbox/astra-toolbox 

If you are using `uv` or `Poetry`, it seems there is an issue with one Tensorflow's dependency which prevents these tools from installing Tensorflow and SimpleITK. You need to manually install them by running: 

``` sh
uv pip install tensorflow
```

and 

``` sh
uv pip install SimpleITK
```

Or their `Poetry` equivalents

## Usage

Once installed, you can use the library in your python code. Example :

``` python
from occipet import reconstruction

reconstruction.MLEM
```

You can also run scripts from the command line. Get more info on the available commands with: 

``` sh
occipet --help
```

