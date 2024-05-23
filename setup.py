import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.mf file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name="caring", # Replace with your own username
    version="0.0.1",
    author="Anonymous",
    author_email="Anonymous@Anonymous.com",
    description="The caring is a tool for discovering latent temporal causal factors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires = [
        "pytorch-lightning==1.2.7",
        "torch==1.11.0",
        "disentanglement-lib==1.4",
        "torchvision",
        "torchaudio",
        "h5py",
        "ipdb",
        "opencv-python",
        "pymunk"
    ],
    tests_require=[
        "pytest"
    ],
)