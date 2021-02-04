# Zennit
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

They may be used by directly importing from `zennit.composites`, or by using their snake-case name as key for
`zennit.composites.COMPOSITES`.
Additionally, there are **Canonizers**, which modify models such that LRP may be applied, if needed.
Currently, there is only one **Canonizer**, which is `MergeBatchNorm`.
There are two versions of `MergeBatchNorm`, `SequentialMergeBatchNorm`, which automatically detects BatchNorm layers
followed by linear layers in sequential networks, and `NamedMergeBatchNorm`, which expects a list of tuples to assign
one or more linear layers to one batch norm layer.

The following is an example how Zennit may be used:
```python
'''A quick example to generate heatmaps for vgg16.'''
import json

import click
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, vgg16_bn, resnet50

from zennit.composites import COMPOSITES
from zennit.image import imsave, CMAPS
from zennit.torchvision import VGGCanonizer, ResNetCanonizer


MODELS = {
    'vgg16': (vgg16, VGGCanonizer),
    'vgg16_bn': (vgg16_bn, VGGCanonizer),
    'resnet50': (resnet50, ResNetCanonizer),
}


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


@click.command()
@click.argument('dataset-root', type=click.Path(file_okay=False))
@click.argument('relevance_format', type=click.Path(dir_okay=False, writable=True))
@click.option('--composite', 'composite_name', type=click.Choice(list(COMPOSITES)), default='epsilon_gamma_box')
@click.option('--model', 'model_name', type=click.Choice(list(MODELS)), default='vgg16_bn')
@click.option('--parameters', type=click.Path(dir_okay=False))
@click.option(
    '--inputs',
    'input_format',
    type=click.Path(dir_okay=False, writable=True),
    help='Input image format string.  {sample} is replaced with the sample index.'
)
@click.option('--batch-size', type=int, default=16)
@click.option('--max-samples', type=int)
@click.option('--n-outputs', type=int, default=1000)
@click.option('--cpu/--gpu', default=True)
@click.option('--shuffle/--no-shuffle', default=False)
@click.option('--cmap', type=click.Choice(list(CMAPS)), default='coldnhot')
@click.option('--level', type=float, default=1.0)
@click.option('--seed', type=int, default=0xDEADBEEF)
def main(
    dataset_root,
    relevance_format,
    composite_name,
    model_name,
    parameters,
    input_format,
    batch_size,
    max_samples,
    n_outputs,
    cpu,
    shuffle,
    cmap,
    level,
    seed
):
    '''Generate heatmaps of an image folder at DATASET_ROOT to files RELEVANCE_FORMAT.
    RELEVANCE_FORMAT is a format string, for which {sample} is replaced with the sample index.
    '''
    # set a manual seed for the RNG
    torch.manual_seed(seed)

    # use the gpu if requested and available, else use the cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')

    # mean and std of ILSVRC2012 as computed for the torchvision models
    norm_fn = BatchNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), device=device)

    # transforms as used for torchvision model evaluation
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
    ])

    # the dataset is a folder containing folders with samples, where each folder corresponds to one label
    dataset = ImageFolder(dataset_root, transform=transform)

    # limit the number of output samples, if requested, by creating a subset
    if max_samples is not None:
        if shuffle:
            indices = sorted(np.random.choice(len(dataset), min(len(dataset), max_samples), replace=False))
        else:
            indices = range(min(len(dataset), max_samples))
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    model = MODELS[model_name][0]()

    # load model parameters if requested; the parameter file may need to be downloaded separately
    if parameters is not None:
        state_dict = torch.load(parameters)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # convenience identity matrix to produce one-hot encodings
    eye = torch.eye(n_outputs, device=device)

    composite_kwargs = {}
    if composite_name == 'epsilon_gamma_box':
        # the maximal input shape, needed for the ZBox rule
        shape = (batch_size, 3, 224, 224)

        # the highest and lowest pixel values for the ZBox rule
        composite_kwargs['low'] = norm_fn(torch.zeros(*shape, device=device))
        composite_kwargs['high'] = norm_fn(torch.ones(*shape, device=device))

    # use torchvision specific canonizers, as supplied in the MODELS dict
    composite_kwargs['canonizers'] = [MODELS[model_name][1]()]

    # create a composite specified by a name; the COMPOSITES dict includes all preset composites provided by zennit.
    composite = COMPOSITES[composite_name](**composite_kwargs)

    # the current sample index for creating file names
    sample = 0

    # create the composite context outside the main loop, such that it canonizers and hooks do not need to be
    # registered and removed for each step.
    with composite.context(model) as modified:
        for data, target in loader:
            # we use data without the normalization applied for visualization, and with the normalization applied as
            # the model input
            data_norm = norm_fn(data.to(device))
            data_norm.requires_grad_()

            # one-hot encoding of the target labels of size (len(target), 1000)
            output_relevance = eye[target]

            out = modified(data_norm)
            # a simple backward pass will accumulate the relevance in data_norm.grad
            torch.autograd.backward((out,), (output_relevance,))

            # sum over the color channel for visualization
            relevance = np.array(data_norm.grad.sum(1).detach().cpu())

            # normalize symmetrically around 0
            amax = relevance.max((1, 2), keepdims=True)
            relevance = (relevance + amax) / 2 / amax

            for n in range(len(data)):
                fname = relevance_format.format(sample=sample + n)
                # zennit.image.imsave will create an appropriate heatmap given a cmap specification
                imsave(fname, relevance[n], vmin=0., vmax=1., level=level, cmap=cmap)
                if input_format is not None:
                    fname = input_format.format(sample=sample + n)
                    # if there are 3 color channels, imsave will not create a heatmap, but instead save the image with
                    # its appropriate colors
                    imsave(fname, data[n])
            sample += len(data)


if __name__ == '__main__':
    main()
```


## Contributing

### Code Style
We use [PEP8](https://www.python.org/dev/peps/pep-0008) with a line-width of 120 characters.
For docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and [`pylint`](https://pypi.org/project/pylint/) for thorough style checks.

### Testing
Tests are written using [pytest](https://pypi.org/project/pylint/) and executed in a separate environment using [tox](https://tox.readthedocs.io/en/latest/).

A full style check and all tests can be run by simply calling `tox` in the repository root.
