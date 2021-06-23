'''A quick example to generate heatmaps for vgg16.'''
import click
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, vgg16_bn, resnet50

from zennit.attribution import Gradient, SmoothGrad, IntegratedGradients
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
@click.option('--absolute-relevance/--no-absolute-relevance', default=False)
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
    cmap,
    level,
    absolute_relevance,
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

    # disable requires_grad for all parameters, we do not need their modified gradients
    for param in model.parameters():
        param.requires_grad = False

    # convenience identity matrix to produce one-hot encodings
    eye = torch.eye(n_outputs, device=device)

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

        # use torchvision specific canonizers, as supplied in the MODELS dict
        composite_kwargs['canonizers'] = [MODELS[model_name][1]()]

        # create a composite specified by a name; the COMPOSITES dict includes all preset composites provided by zennit.
        composite = COMPOSITES[composite_name](**composite_kwargs)

    # create an attributor, given the ATTRIBUTORS dict given above. If composite is None, the gradient will not be
    # modified for the attribution
    attributor = ATTRIBUTORS[attributor_name](model, composite)

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

            # one-hot encoding of the target labels of size (len(target), 1000)
            output_relevance = eye[target]

            # this will compute the modified gradient of model, with the on
            output, relevance = attributor(data_norm, output_relevance)

            # sum over the color channel for visualization
            relevance = np.array(relevance.sum(1).detach().cpu())

            if absolute_relevance:
                # use the absolute relevance, normalized between 0. and 1.
                relevance = np.abs(relevance)
                relevance /= relevance.max((1, 2), keepdims=True)
            else:
                # normalize symmetrically around 0, then normalize between 0. and 1.
                amax = relevance.max((1, 2), keepdims=True)
                relevance = (relevance + amax) / 2 / amax

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
