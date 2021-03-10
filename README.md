# Zennit
![Zennit-Logo](share/img/zennit.png)


Zennit (**Z**ennit **e**xplains **n**eural **n**etworks **i**n **t**orch)
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
**Composites** are a way of choosing the right hook for the right layer.
In addition to the abstract **NameMapComposite**, which assigns hooks to layers by name, and **LayerMapComposite**,
which assigns hooks to layers based on their Type, there exist explicit Composites, which currently are
* EpsilonGammaBox (ZBox in input, epsilon in dense, Gamma 0.25 in convolutions)
* EpsilonPlus (PresetA in iNNvestigate)
* EpsilonPlusFlat (PresetAFlat in iNNvestigate)
* EpsilonAlpha2Beta1 (PresetB in iNNvestigate)
* EpsilonAlpha2Beta1Flat (PresetBFlat in iNNvestigate).

They may be used by directly importing from `zennit.composites`, or by using
their snake-case name as key for `zennit.composites.COMPOSITES`. Additionally,
there are **Canonizers**, which modify models such that LRP may be applied, if
needed. Currently, there are `MergeBatchNorm`, `AttributeCanonizer` and
`CompositeCanonizer`. There are two versions of the abstract `MergeBatchNorm`,
`SequentialMergeBatchNorm`, which automatically detects BatchNorm layers
followed by linear layers in sequential networks, and `NamedMergeBatchNorm`,
which expects a list of tuples to assign one or more linear layers to one batch
norm layer. `AttributeCanonizer` temporarily overwrites attributes of
applicable modules, e.g. for ResNet50, the forward function (attribute) of the
Bottleneck modules is overwritten to handle the residual connection.

## Example
Prepare the data needed for the example (requires cURL and (magic-)file):
```shell
$ mkdir -p share/params share/data share/results
$ bash share/scripts/subimagenet.sh --n-total 8 --wnid n02814860 --output share/data/tiny_imagenet
$ curl -o share/params/vgg16-397923af.pth 'https://download.pytorch.org/models/vgg16-397923af.pth'
```
This creates the needed directories and downloads the pre-trained vgg16 parameters and a tiny subset of imagenet with the required label-directory structure and 8 samples of class *beacon* (n02814860).

The example at `share/example/feed_forward.py` may then be run using:
```shell
$ python share/example/feed_forward.py \
    share/data/tiny_imagenet \
    'share/results/vgg16_epsilon_gamma_box_{sample:02d}.png' \
    --inputs 'share/results/vgg16_input_{sample:02d}.png' \
    --parameters share/params/vgg16-397923af.pth \
    --model vgg16 \
    --composite epsilon_gamma_box
```
which computes the lrp heatmaps according to the `epsilon_gamma_box` rule and stores them in `share/results`, along with the respective input images.

The following is a slightly modified exerpt of `share/example/feed_forward.py`:
```python
...
    # the maximal input shape, needed for the ZBox rule
    shape = (batch_size, 3, 224, 224)

    composite_kwargs = {
        'low': norm_fn(torch.zeros(*shape, device=device)),  # the highest and ...
        'high': norm_fn(torch.ones(*shape, device=device)),  # the lowest pixel value for ZBox
        'canonizers': [VGG16Canonizer()]  # the torchvision specific vgg16 canonizer
    }

    # create a composite specified by a name; the COMPOSITES dict includes all preset composites
    # provided by zennit.
    composite = COMPOSITES['epsilon_gamma_box'](**composite_kwargs)

    # create the composite context outside the main loop, such that the canonizers and hooks do not
    # need to be registered and removed for each step.
    with composite.context(model) as modified_model:
        for data, target in loader:
            # we use data without the normalization applied for visualization, and with the
            # normalization applied as the model input
            data_norm = norm_fn(data.to(device))
            data_norm.requires_grad_()

            # one-hot encoding of the target labels of size (len(target), 1000)
            output_relevance = torch.eye(n_outputs, device=device)[target]

            out = modified_model(data_norm)
            # a simple backward pass will accumulate the relevance in data_norm.grad
            torch.autograd.backward((out,), (output_relevance,))
...
```


## Contributing

### Code Style
We use [PEP8](https://www.python.org/dev/peps/pep-0008) with a line-width of 120 characters.
For docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and [`pylint`](https://pypi.org/project/pylint/) for thorough style checks.

### Testing
Tests are written using [pytest](https://pypi.org/project/pylint/) and executed in a separate environment using [tox](https://tox.readthedocs.io/en/latest/).

A full style check and all tests can be run by simply calling `tox` in the repository root.
