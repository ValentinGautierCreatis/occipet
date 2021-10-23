# Occipet

Occipet is a python package I have built to have several customizable tools to work with medical images. 
It is a library that can be used as a building block in other python scripts. It also exposes several 
scripts that can be called from the command line thanks to [Typer](https://typer.tiangolo.com/).

## Installation 

First, clone this repository:

``` sh
git clone git@github.com:ValentinGautierCreatis/occipet.git
```

Then you have to install dependencies. I recommend doing it in a virtual environment (conda, virtualenv, pyenv, ...)

``` sh
pip install -r requirements.txt
```

Finally, just install the library by running the following at the root of the repository:

``` sh
pip install .
```

You can install it in development mode. Changes in the source files while be applied without having to install 
the package again. To do so, run the following :

``` sh
pip install -e .
```

## Usage

You can use the library in your python code. Example :

``` python
from occipet import reconstruction

reconstruction.MLEM
```

You can also run scripts from the command line. Get more info on the available commands with: 

``` sh
occipet --help
```

