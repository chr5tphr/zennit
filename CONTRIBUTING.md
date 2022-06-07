# Contributing Guide for Zennit

Thank you for your interest in contributing to Zennit!

If you would like to fix a bug or add a feature, please write an issue before submitting a pull request.


## Git
We use a linear git-history, where each commit contains a full feature/bug fix,
such that each commit represents an executable version.
The commit message contains a subject followed by an empty line, followed by a detailed description, similar to the following:

```
Category: Short subject describing changes (50 characters or less)

- detailed description, wrapped at 72 characters
- bullet points or sentences are okay
- all changes should be documented and explained
- valid categories are, for example:
    - `Docs` for documentation
    - `Tests` for tests
    - `Composites` for changes in composites
    - `Core` for core changes
    - `Package` for package-related changes, e.g. in setup.py
```

We recommend to not use `-m` for committing, as this often results in very short commit messages.

## Code Style
We use [PEP8](https://www.python.org/dev/peps/pep-0008) with a line-width of 120 characters. For
docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and
[`pylint`](https://pypi.org/project/pylint/) for thorough style checks.

## Testing
Tests are written using [Pytest](https://docs.pytest.org) and executed
in a separate environment using [Tox](https://tox.readthedocs.io/en/latest/).

A full style check and all tests can be run by simply calling `tox` in the repository root.

If you add a new feature, please also include appropriate tests to verify its intended functionality.
We try to keep the code coverage close to 100%.

## Documentation
The documentation uses [Sphinx](https://www.sphinx-doc.org). It can be built at
`docs/build` using the respective Tox environment with `tox -e docs`. To rebuild the full
documentation, `tox -e docs -- -aE` can be used.

The API-documentation is generated from the numpycodestyle docstring of respective modules/classes/functions.

### Tutorials
Tutorials are written as Jupyter notebooks in order to execute them using
[Binder](https://mybinder.org/) or [Google
Colab](https://colab.research.google.com/).
They are found at [`docs/source/tutorial`](docs/source/tutorial).
Their output should be empty when committing, as they will be executed when
building the documentation.
To reduce the building time of the documentation, their execution time should
be kept short, i.e. large files like model parameters should not be downloaded
automatically.
To include parameter files for users, include a comment which describes how to
use the full model/data, and provide the necessary code in a comment or an if-condition
which always evaluates to `False`.

## Continuous Integration
Linting, tests and the documentation are all checked using a Github Actions
workflow which executes the appropriate tox environments.
