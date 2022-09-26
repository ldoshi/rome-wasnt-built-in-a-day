#!/usr/bin/env python
from setuptools import setup, find_packages

from bridger import __version__

setup(
    name="bridger",
    version=__version__,
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    scripts=["bin/bridge_builder.py"],
    install_requires=[
        "ipython>=7.21.0",
        "gym>=0.17.3",
        "gym-bridges>=0.0.1",
        "numpy>=1.18.5",
        "matplotlib>=3.3.2",
        "pytorch-lightning==1.6.3",
        "torch>=1.8.0",
    ],
    # TODO: add metadata kwargs if uploading to PyPI
)
