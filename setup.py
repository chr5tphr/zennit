#!/usr/bin/env python3
import re
from setuptools import setup, find_packages
from subprocess import run, CalledProcessError


def get_long_description(project_path):
    '''Fetch the README contents and replace relative links with absolute ones
    pointing to github for correct behaviour on PyPI.
    '''
    try:
        revision = run(
            ['git', 'describe', '--tags'],
            capture_output=True,
            check=True,
            text=True
        ).stdout[:-1]
    except CalledProcessError:
        try:
            with open('PKG-INFO', 'r') as fd:
                body = fd.read().partition('\n\n')[2]
            if body:
                return body
        except FileNotFoundError:
            revision = 'master'

    with open('README.md', 'r', encoding='utf-8') as fd:
        long_description = fd.read()

    link_root = {
        '': f'https://github.com/{project_path}/blob',
        '!': f'https://raw.githubusercontent.com/{project_path}',
    }

    def replace(mobj):
        return f'{mobj[1]}[{mobj[2]}]({link_root[mobj[1]]}/{revision}/{mobj[3]})'

    link_rexp = re.compile(r'(!?)\[([^\]]*)\]\((?!https?://|/)([^\)]+)\)')
    return link_rexp.sub(replace, long_description)


setup(
    name='zennit',
    use_scm_version=True,
    author='chrstphr',
    author_email='zennit@j0d.de',
    description='Attribution of Neural Networks using PyTorch',
    long_description=get_long_description('chr5tphr/zennit'),
    long_description_content_type='text/markdown',
    url='https://github.com/chr5tphr/zennit',
    packages=find_packages(where='src', include=['zennit*']),
    package_dir={'': 'src'},
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
    extras_require={
        'docs': [
            'sphinx-copybutton>=0.4.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinxcontrib.datatemplates>=0.9.0',
            'sphinxcontrib.bibtex>=2.4.1',
            'nbsphinx>=0.8.8',
            'ipykernel>=6.13.0',
        ],
        'tests': [
            'pytest',
            'pytest-cov',
        ]
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)
