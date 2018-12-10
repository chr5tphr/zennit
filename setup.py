from setuptools import setup, find_packages
import setuptools

VERSION = '0.1'

def run_setup():
    requires = [
        'numpy>=1.15.3',
        'Pillow>=5.3.0',
        'six>=1.11.0',
        'torch>=0.4.1.post2',
        'torchvision>=0.2.1',
    ]
    setup(
        name="zennit",
        version=VERSION,
        packages=find_packages(),
        install_requires=requires)

run_setup()
