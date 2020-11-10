# Zennit
Zennit (__Z__ennit __e__xplains __n__eural __n__etworks __i__n __t__orch)
is a high-level framework in Python using PyTorch for explaining/exploring neural networks.
Its design philosophy is intended to provide high customizability and integration as a standardized solution
for applying LRP-based attribution methods in research.


Zennit is currently under development and has not yet reached a stable state.
Interfaces may change suddenly and without warning, so please be careful when attempting to use Zennit in its current
state.

## Usage
An example can be found in `share/example/feed_forward.py`.
Currently, only feed-forward type models are supported.

At its heart, Zennit registers hooks at PyTorch's Module level, to modify the backward pass to produce LRP
attributions (instead of the usual gradient).
All rules are implemented as hooks (`zennit/rules.py`) and most use the basic `LinearHook` (`zennit/core.py`).
__Composites__ are a way of choosing the right hook for the right layer.
In addition to the abstract __NameMapComposite__, which assigns hooks to layers by name, and __LayerMapComposite__,
which assigns hooks to layers based on their Type, there exist explicit Composites, which currently are
* EpsilonGammaBox (ZBox in input, epsilon in dense, Gamma 0.25 in convolutions)
* EpsilonPlus (PresetA in iNNvestigate)
* EpsilonPlusFlat (PresetAFlat in iNNvestigate)
* EpsilonAlpha2Beta1 (PresetB in iNNvestigate)
* EpsilonAlpha2Beta1Flat (PresetBFlat in iNNvestigate).

They may be used by directly importing from `zennit.composites`, or by using their snake-case name as key for
`zennit.composites.COMPOSITES`.
Additionally, there are __Canonizers__, which modify models such that LRP may be applied, if needed.
Currently, there is only one __Canonizer__, which is `MergeBatchNorm`.
There are two versions of `MergeBatchNorm`, `SequentialMergeBatchNorm`, which automatically detects BatchNorm layers
followed by linear layers in sequential networks, and `NamedMergeBatchNorm`, which expects a list of tuples to assign
one or more linear layers to one batch norm layer.

The following is a minimal example how Zennit may be used:
```python
import torch
from torchvision.models import vgg16_bn

from zennit.composite import EpsilonGammaBox
from zennit.canonizers import SequentialBatchNorm


shape = (16, 3, 224, 224)

# generate some random data
data = torch.randn(*shape, requires_grad=True)
target = torch.randint(1000)

# EpsilonGammaBox uses the ZBox rule, which needs minimum and maximum values
low = torch.full(shape, -3)
high = torch.full(shape, 3)

# vgg16_bn uses BatchNorm, so we need a canonizer
canonizers = [SequentialMergeBatchNorm()]

composite = EpsilonGammaBox(low, high, canonizers)

# the composite can apply canonizers in a context, which will be fully cleaned
# up after exiting the context
with composite.context(model) as modified:
    # one-hot tensor of target
    output_relevance = torch.eye(n_outputs, device=device)[target]
    out = modified(data)
    torch.autograd.backward((out,), (output_relevance,))

# the attribution will be stored in the gradient's place
relevance = data_norm.grad.sum(1)
imsave('attribution.png', relevance, grid=True, cmap='coldnhot')
```


## Contributing

### Code Style
We use [PEP8](https://www.python.org/dev/peps/pep-0008) with a line-width of 120 characters.
For docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and [`pylint`](https://pypi.org/project/pylint/) for thorough style checks.

### Testing
Tests are written using [pytest](https://pypi.org/project/pylint/) and executed in a separate environment using [tox](https://tox.readthedocs.io/en/latest/).

A full style check and all tests can be run by simply calling `tox` in the repository root.
