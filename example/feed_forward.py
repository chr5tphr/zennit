'''A quick example to generate heatmaps for vgg16.'''
import json

import click
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, vgg16_bn

from zennit.composites import COMPOSITES
from zennit.image import imsave, gridify
from zennit.canonizers import SequentialMergeBatchNorm, NamedMergeBatchNorm


MODELS = {
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
}


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


@click.command()
@click.argument('dataset-root', type=click.Path(file_okay=False))
@click.argument('output', type=click.Path(dir_okay=False, writable=True))
@click.option('--composite', 'composite_name', type=click.Choice(list(COMPOSITES)), default='epsilon_gamma_box')
@click.option('--model', 'model_name', type=click.Choice(list(MODELS)), default='vgg16_bn')
@click.option('--parameters', type=click.Path(dir_okay=False))
@click.option('--inputs', type=click.Path(dir_okay=False, writable=True))
@click.option('--batch-size', type=int, default=16)
@click.option('--n-outputs', type=int, default=1000)
@click.option('--merge-map', 'merge_map_file', type=click.Path(dir_okay=False))
@click.option('--cpu/--gpu', default=True)
def main(
    dataset_root,
    output,
    composite_name,
    model_name,
    parameters,
    inputs,
    batch_size,
    n_outputs,
    merge_map_file,
    cpu
):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')

    norm_fn = BatchNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), device=device)

    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
    ])
    dataset = ImageFolder(dataset_root, transform=transform)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    model = MODELS[model_name]()

    if parameters is not None:
        state_dict = torch.load(parameters)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    eye = torch.eye(n_outputs, device=device)

    shape = (batch_size, 3, 224, 224)

    composite_kwargs = {}
    if composite_name == 'epsilon_gamma_box':
        composite_kwargs['low'] = norm_fn(torch.zeros(*shape, device=device))
        composite_kwargs['high'] = norm_fn(torch.ones(*shape, device=device))

    if merge_map_file is not None:
        with open(merge_map_file, 'r') as fd:
            merge_map = json.load(fd)
        composite_kwargs['canonizers'] = [NamedMergeBatchNorm(merge_map)]
    else:
        composite_kwargs['canonizers'] = [SequentialMergeBatchNorm()]

    composite = COMPOSITES[composite_name](**composite_kwargs)

    with composite.context(model) as modified:
        data, target = next(loader)
        data = data.to(device)
        data_norm = norm_fn(data)
        data_norm.requires_grad_()

        output_relevance = eye[target]

        out = modified(data_norm)
        torch.autograd.backward((out,), (output_relevance,))

    # visualize heatmaps
    relevance = np.array(data_norm.grad.sum(1).detach().cpu())
    amax = relevance.max((1, 2), keepdims=True)
    relevance = (relevance + amax) / 2 / amax
    grid = gridify(relevance, fill_value=0.5)
    imsave(output, grid, vmin=0., vmax=1., level=1.0, cmap='coldnhot')

    if inputs is not None:
        imsave(inputs, data.detach().cpu(), grid=True)


if __name__ == '__main__':
    main()
