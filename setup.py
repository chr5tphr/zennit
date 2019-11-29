from setuptools import setup, find_packages
import setuptools

VERSION = '0.1'

def run_setup():
    setup(
        name="zennit",
        version=VERSION,
        packages=find_packages(),
        install_requires=[
            'numpy',
            'Pillow',
            'torch',
            'torchvision',
        ],
    )

run_setup()
