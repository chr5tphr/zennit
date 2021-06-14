#!/usr/bin/env python3
from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fd:
    long_description = fd.read()


setup(
    name='zennit',
    use_scm_version=True,
    author='chrstphr',
    author_email='zennit@j0d.de',
    description='Attribution of Neural Networks using PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chr5tphr/zennit',
    packages=find_packages(include=['zennit*']),
    install_requires=[
        'click',
        'numpy',
        'Pillow',
        'torch>=1.7.0',
        'torchvision',
    ],
    setup_requires=[
        'setuptools_scm',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)
