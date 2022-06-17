import setuptools
import glob

def requirements():
    """List requirements from requirement.txt file
    """
    if glob.glob('requirements.txt'):
        with open('requirements.txt') as requirement_file:
            return [req.strip() for req in requirement_file.readlines()]
    else:
        return []
setuptools.setup(
    name="occipet",
    version="0.0.1",
    author="Valentin Gautier",
    author_email="valentin.gautier@creatis.insa-lyon.fr",
    description="Imaging for PET et MRI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    #install_requires=requirements(),
    entry_points = {
        "console_scripts": [
            "occipet=occipet.app:app",
            "deep_learning=deep_learning.app:app"
        ]
    },
    python_requires=">=3.6",
)
