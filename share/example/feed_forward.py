'''A quick example to generate heatmaps for vgg16.'''
import os
from functools import partial

import click
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, vgg16_bn, resnet50

from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients, Occlusion
from zennit.composites import COMPOSITES
from zennit.image import imsave, CMAPS
from zennit.torchvision import VGGCanonizer, ResNetCanonizer


MODELS = {
    'vgg16': (vgg16, VGGCanonizer),
    'vgg16_bn': (vgg16_bn, VGGCanonizer),
    'resnet50': (resnet50, ResNetCanonizer),
}

ATTRIBUTORS = {
    'gradient': Gradient,
    'smoothgrad': SmoothGrad,
    'integrads': IntegratedGradients,
    'occlusion': Occlusion,
}


class BatchNormalize:
    def __init__(self, mean, std, device=None):
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class AllowEmptyClassImageFolder(ImageFolder):
    '''Subclass of ImageFolder, which only finds non-empty classes, but with their correct indices given other empty
    classes. This counter-acts the changes in torchvision 0.10.0, in which DatasetFolder does not allow empty classes
    anymore by default. Versions before 0.10.0 do not expose `find_classes`, and thus this change does not change the
    functionality of `ImageFolder` in earlier versions.
    '''
    def find_classes(self, directory):
        with os.scandir(directory) as scanit:
            class_info = sorted((entry.name, len(list(os.scandir(entry.path)))) for entry in scanit if entry.is_dir())
        class_to_idx = {class_name: index for index, (class_name, n_members) in enumerate(class_info) if n_members}
        if not class_to_idx:
            raise FileNotFoundError(f'No non-empty classes found in \'{directory}\'.')
        return list(class_to_idx), class_to_idx


@click.command()
@click.argument('dataset-root', type=click.Path(file_okay=False))
@click.argument('relevance_format', type=click.Path(dir_okay=False, writable=True))
@click.option('--attributor', 'attributor_name', type=click.Choice(list(ATTRIBUTORS)), default='gradient')
@click.option('--composite', 'composite_name', type=click.Choice(list(COMPOSITES)))
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
@click.option('--with-bias/--no-bias', default=True)
@click.option('--relevance-norm', type=click.Choice(['symmetric', 'absolute', 'unaligned']), default='symmetric')
@click.option('--cmap', type=click.Choice(list(CMAPS)), default='coldnhot')
@click.option('--level', type=float, default=1.0)
@click.option('--seed', type=int, default=0xDEADBEEF)
def main(
    dataset_root,
    relevance_format,
    attributor_name,
    composite_name,
    model_name,
    parameters,
    input_format,
    batch_size,
    max_samples,
    n_outputs,
    cpu,
    shuffle,
    with_bias,
    cmap,
    level,
    relevance_norm,
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
    dataset = AllowEmptyClassImageFolder(dataset_root, transform=transform)

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

    # disable requires_grad for all parameters, we do not need their modified gradients
    for param in model.parameters():
        param.requires_grad = False

    # convenience identity matrix to produce one-hot encodings
    eye = torch.eye(n_outputs, device=device)

    # function to compute output relevance given the function output and a target
    def attr_output_fn(output, target):
        # output times one-hot encoding of the target labels of size (len(target), 1000)
        return output * eye[target]

    # create a composite if composite_name was set, otherwise we do not use a composite
    composite = None
    if composite_name is not None:
        composite_kwargs = {}
        if composite_name == 'epsilon_gamma_box':
            # the maximal input shape, needed for the ZBox rule
            shape = (batch_size, 3, 224, 224)

            # the highest and lowest pixel values for the ZBox rule
            composite_kwargs['low'] = norm_fn(torch.zeros(*shape, device=device))
            composite_kwargs['high'] = norm_fn(torch.ones(*shape, device=device))

        # provide the name 'bias' in zero_params if no bias should be used to compute the relevance
        if not with_bias and composite_name in [
            'epsilon_gamma_box',
            'epsilon_plus',
            'epsilon_alpha2_beta1',
            'epsilon_plus_flat',
            'epsilon_alpha2_beta1_flat',
            'excitation_backprop',
        ]:
            composite_kwargs['zero_params'] = ['bias']

        # use torchvision specific canonizers, as supplied in the MODELS dict
        composite_kwargs['canonizers'] = [MODELS[model_name][1]()]

        # create a composite specified by a name; the COMPOSITES dict includes all preset composites provided by zennit.
        composite = COMPOSITES[composite_name](**composite_kwargs)

    # specify some attributor-specific arguments
    attributor_kwargs = {
        'smoothgrad': {'noise_level': 0.1, 'n_iter': 20},
        'integrads': {'n_iter': 20},
        'occlusion': {'window': (56, 56), 'stride': (28, 28)},
    }.get(attributor_name, {})

    # create an attributor, given the ATTRIBUTORS dict given above. If composite is None, the gradient will not be
    # modified for the attribution
    attributor = ATTRIBUTORS[attributor_name](model, composite, **attributor_kwargs)

    # the current sample index for creating file names
    sample_index = 0

    # the accuracy
    accuracy = 0.

    # enter the attributor context outside the data loader loop, such that its canonizers and hooks do not need to be
    # registered and removed for each step. This registers the composite (and applies the canonizer) to the model
    # within the with-statement
    with attributor:
        for data, target in loader:
            # we use data without the normalization applied for visualization, and with the normalization applied as
            # the model input
            data_norm = norm_fn(data.to(device))

            # create output relevance function of output with fixed target
            output_relevance = partial(attr_output_fn, target=target)

            # this will compute the modified gradient of model, where the output relevance is chosen by the as the
            # model's output for the ground-truth label index
            output, relevance = attributor(data_norm, output_relevance)

            # sum over the color channel for visualization
            relevance = np.array(relevance.sum(1).detach().cpu())

            # normalize between 0. and 1. given the specified strategy
            if relevance_norm == 'symmetric':
                # 0-aligned symmetric relevance, negative and positive can be compared, the original 0. becomes 0.5
                amax = np.abs(relevance).max((1, 2), keepdims=True)
                relevance = (relevance + amax) / 2 / amax
            elif relevance_norm == 'absolute':
                # 0-aligned absolute relevance, only the amplitude of relevance matters, the original 0. becomes 0.
                relevance = np.abs(relevance)
                relevance /= relevance.max((1, 2), keepdims=True)
            elif relevance_norm == 'unaligned':
                # do not align, the original minimum value becomes 0., the original maximum becomes 1.
                rmin = relevance.min((1, 2), keepdims=True)
                rmax = relevance.max((1, 2), keepdims=True)
                relevance = (relevance - rmin) / (rmax - rmin)

            for n in range(len(data)):
                fname = relevance_format.format(sample=sample_index + n)
                # zennit.image.imsave will create an appropriate heatmap given a cmap specification
                imsave(fname, relevance[n], vmin=0., vmax=1., level=level, cmap=cmap)
                if input_format is not None:
                    fname = input_format.format(sample=sample_index + n)
                    # if there are 3 color channels, imsave will not create a heatmap, but instead save the image with
                    # its appropriate colors
                    imsave(fname, data[n])
            sample_index += len(data)

            # update the accuracy
            accuracy += (output.argmax(1) == target).sum().detach().cpu().item()

    accuracy /= len(dataset)
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    main()
