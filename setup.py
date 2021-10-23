import setuptools

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
    entry_points = {
        "console_scripts": [
            "occipet=occipet.app:app"
        ]
    },
    python_requires=">=3.6",
)
