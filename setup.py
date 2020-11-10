#!/usr/bin/env python3
from setuptools import setup

setup(
    name="zennit",
    use_scm_version=True,
    packages=['zennit'],
    install_requires=[
        'numpy',
        'Pillow',
        'torch',
    ],
    setup_requires=[
        'setuptools_scm',
    ],
)
