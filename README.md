# Zennit
![Zennit-Logo](share/img/zennit.png)

[![Documentation Status](https://readthedocs.org/projects/zennit/badge/?version=latest)](https://zennit.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/chr5tphr/zennit/actions/workflows/tests.yml/badge.svg)](https://github.com/chr5tphr/zennit/actions/workflows/tests.yml)
[![PyPI Version](https://img.shields.io/pypi/v/zennit)](https://pypi.org/project/zennit/)
[![License](https://img.shields.io/pypi/l/zennit)](https://github.com/chr5tphr/zennit/blob/master/COPYING.LESSER)

Zennit (**Z**ennit **e**xplains **n**eural **n**etworks **i**n **t**orch) is a
high-level framework in Python using Pytorch for explaining/exploring neural
networks. Its design philosophy is intended to provide high customizability and
integration as a standardized solution for applying rule-based attribution
methods in research, with a strong focus on Layerwise Relevance Propagation
(LRP). Zennit strictly requires models to use Pytorch's `torch.nn.Module`
structure (including activation functions).

Zennit is currently under active development, but should be mostly stable.

If you find Zennit useful for your research, please consider citing our related
[paper](https://arxiv.org/abs/2106.13200):
```
@article{anders2021software,
      author  = {Anders, Christopher J. and
                 Neumann, David and
                 Samek, Wojciech and
                 Müller, Klaus-Robert and
                 Lapuschkin, Sebastian},
      title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
      journal = {CoRR},
      volume  = {abs/2106.13200},
      year    = {2021},
}
```

## Documentation
The latest documentation is hosted at
[zennit.readthedocs.io](https://zennit.readthedocs.io/en/latest/).

## Install

To install directly from PyPI using pip, use:
```shell
$ pip install zennit
```

Alternatively, install from a manually cloned repository to try out the examples:
```shell
$ git clone https://github.com/chr5tphr/zennit.git
$ pip install ./zennit
```

## Usage
At its heart, Zennit registers hooks at Pytorch's Module level, to modify the
backward pass to produce rule-based attributions like LRP (instead of the usual
gradient). All rules are implemented as hooks
([`zennit/rules.py`](zennit/rules.py)) and most use the LRP basis
`BasicHook` ([`zennit/core.py`](zennit/core.py)).

**Composites** ([`zennit/composites.py`](zennit/composites.py)) are a way of
choosing the right hook for the right layer. In addition to the abstract
**NameMapComposite**, which assigns hooks to layers by name, and
**LayerMapComposite**, which assigns hooks to layers based on their Type, there
exist explicit **Composites**, some of which are `EpsilonGammaBox` (`ZBox` in
input, `Epsilon` in dense, `Gamma` in convolutions) or `EpsilonPlus` (`Epsilon`
in dense, `ZPlus` in convolutions). All composites may be used by directly
importing from `zennit.composites`, or by using their snake-case name as key
for `zennit.composites.COMPOSITES`.

**Canonizers** ([`zennit/canonizers.py`](zennit/canonizers.py)) temporarily
transform models into a canonical form, if required, like
`SequentialMergeBatchNorm`, which automatically detects and merges BatchNorm
layers followed by linear layers in sequential networks, or
`AttributeCanonizer`, which temporarily overwrites attributes of applicable
modules, e.g. to handle the residual connection in ResNet-Bottleneck modules.

**Attributors** ([`zennit/attribution.py`](zennit/attribution.py)) directly
execute the necessary steps to apply certain attribution methods, like the
simple `Gradient`, `SmoothGrad` or `Occlusion`. An optional **Composite** may
be passed, which will be applied during the **Attributor**'s execution to
compute the modified gradient, or hybrid methods.

Using all of these components, an LRP-type attribution for VGG16 with
batch-norm layers with respect to label 0 may be computed using:

```python
import torch
from torchvision.models import vgg16_bn

from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient


data = torch.randn(1, 3, 224, 224)
model = vgg16_bn()

canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)

with Gradient(model=model, composite=composite) as attributor:
    out, relevance = attributor(data, torch.eye(1000)[[0]])
```

For more details and examples, have a look at our
[**documentation**](https://zennit.readthedocs.io/en/latest/).

## Example
This example demonstrates how the script at
[`share/example/feed_forward.py`](share/example/feed_forward.py) can be used to
generate attribution heatmaps for VGG16.
It requires bash, cURL and (magic-)file.

Create a virtual environment, install Zennit and download the example scripts:
```shell
$ mkdir zennit-example
$ cd zennit-example
$ python -m venv .venv
$ .venv/bin/pip install zennit
$ curl -o feed_forward.py \
    'https://raw.githubusercontent.com/chr5tphr/zennit/master/share/example/feed_forward.py'
$ curl -o download-lighthouses.sh \
    'https://raw.githubusercontent.com/chr5tphr/zennit/master/share/scripts/download-lighthouses.sh'
```

Prepare the data needed for the example :
```shell
$ mkdir params data results
$ bash download-lighthouses.sh --output data/lighthouses
$ curl -o params/vgg16-397923af.pth 'https://download.pytorch.org/models/vgg16-397923af.pth'
```
This creates the needed directories and downloads the pre-trained VGG16 parameters and 8 images of light houses from Wikimedia Commons into the required label-directory structure for the Imagenet dataset in Pytorch.

The `feed_forward.py` example may then be run using:
```shell
$ .venv/bin/python feed_forward.py \
    data/lighthouses \
    'results/vgg16_epsilon_gamma_box_{sample:02d}.png' \
    --inputs 'results/vgg16_input_{sample:02d}.png' \
    --parameters params/vgg16-397923af.pth \
    --model vgg16 \
    --composite epsilon_gamma_box \
    --relevance-norm symmetric \
    --cmap coldnhot
```
which computes the LRP heatmaps according to the `epsilon_gamma_box` rule and
stores them in `results`, along with the respective input images. Other
possible composites that can be passed to `--composites` are, e.g.,
`epsilon_plus`, `epsilon_alpha2_beta1_flat`, `guided_backprop`,
`excitation_backprop`.

The resulting heatmaps may look like the following:
![beacon heatmaps](share/img/beacon_vgg16_epsilon_gamma_box.png)

Alternatively, heatmaps for SmoothGrad with absolute relevances may be computed
by omitting `--composite` and supplying `--attributor`:
```shell
$ .venv/bin/python feed_forward.py \
    data/lighthouses \
    'results/vgg16_smoothgrad_{sample:02d}.png' \
    --inputs 'results/vgg16_input_{sample:02d}.png' \
    --parameters params/vgg16-397923af.pth \
    --model vgg16 \
    --attributor smoothgrad \
    --relevance-norm absolute \
    --cmap hot
```
For Integrated Gradients, `--attributor integrads` may be provided.

Heatmaps for Occlusion Analysis with unaligned relevances may be computed by
executing:
```shell
$ .venv/bin/python feed_forward.py \
    data/lighthouses \
    'results/vgg16_occlusion_{sample:02d}.png' \
    --inputs 'results/vgg16_input_{sample:02d}.png' \
    --parameters params/vgg16-397923af.pth \
    --model vgg16 \
    --attributor occlusion \
    --relevance-norm unaligned \
    --cmap hot
```

## Example Heatmaps
Heatmaps of various attribution methods for VGG16 and ResNet50, all generated using
[`share/example/feed_forward.py`](share/example/feed_forward.py), can be found below.

<details>
  <summary>Heatmaps for VGG16</summary>

  ![vgg16 heatmaps](share/img/beacon_vgg16_various.webp)
</details>

<details>
  <summary>Heatmaps for ResNet50</summary>

  ![resnet50 heatmaps](share/img/beacon_resnet50_various.webp)
</details>

## Contributing

### Code Style
We use [PEP8](https://www.python.org/dev/peps/pep-0008) with a line-width of 120 characters. For
docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and
[`pylint`](https://pypi.org/project/pylint/) for thorough style checks.

### Testing
Tests are written using [Pytest](https://docs.pytest.org) and executed
in a separate environment using [Tox](https://tox.readthedocs.io/en/latest/).

A full style check and all tests can be run by simply calling `tox` in the repository root.

### Documentation
The documentation is written using [Sphinx](https://www.sphinx-doc.org). It can be built at
`docs/build` using the respective Tox environment with `tox -e docs`. To rebuild the full
documentation, `tox -e docs -- -aE` can be used.
