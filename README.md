# Zennit
Zennit (__Z__ennit __e__xplains __n__eural __n__etworks __i__n __t__orch)
is a high-level framework in Python using PyTorch for explaining/exploring neural networks.

This Repository serves as the initial core code, and will eventually evolve to a fully fledged Framework.

## Code Style
We use [PEP8](https://www.python.org/dev/peps/pep-0008) with a line-width of 120 characters.
For docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and [`pylint`](https://pypi.org/project/pylint/) for thorough style checks.

## Testing
Tests are written using [pytest](https://pypi.org/project/pylint/) and executed in a separate environment using [tox](https://tox.readthedocs.io/en/latest/).

A full style check and all tests can be run by simply calling `tox` in the repository root.
