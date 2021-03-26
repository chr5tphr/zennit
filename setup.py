#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="zennit",
    use_scm_version=True,
    packages=find_packages(include=['zennit*']),
    install_requires=[
        'numpy',
        'Pillow',
        'torch>=1.7.0',
    ],
    setup_requires=[
        'setuptools_scm',
    ],
)
