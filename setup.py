"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=find_packages(include=("src*", "flash_rl*", "scripts*")),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
