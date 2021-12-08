'''Configuration and fixtures for testing'''
from itertools import product

import pytest
import torch
from torch.nn import Conv1d, ConvTranspose1d, Linear
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import Conv3d, ConvTranspose3d
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


def prodict(**kwargs):
    return [dict(zip(kwargs, val)) for val in product(*kwargs.values())]


@pytest.fixture(scope='session', params=[
    0xdeadbeef,
    0xd0c0ffee,
    *[pytest.param(seed, marks=pytest.mark.extended) for seed in [
        0xc001bee5, 0xc01dfee7, 0xbe577001, 0xca7b0075, 0x1057b0a7, 0x900ddeed
    ]]
])
def rng(request):
    return torch.manual_seed(request.param)


@pytest.fixture(scope='session', params=[
    *product(
        [Linear],
        prodict(in_features=[16], out_features=[16], bias=[True, False]),
    ),
    *product(
        [Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
        prodict(in_channels=[1, 3], out_channels=[1, 3], kernel_size=[2, 3], bias=[True, False]),
    ),
])
def module_linear(rng, request):
    module_type, kwargs = request.param
    return module_type(**kwargs).eval()


@pytest.fixture(scope='session')
def module_batchnorm(module_linear):
    module_map = [
        ((Linear, Conv1d, ConvTranspose1d), BatchNorm1d),
        ((Conv2d, ConvTranspose2d), BatchNorm2d),
        ((Conv3d, ConvTranspose3d), BatchNorm3d),
    ]
    feature_index_map = [
        ((Linear, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d), 1),
        ((Conv1d, Conv2d, Conv3d), 0),
    ]

    batchnorm_type = None
    for types, target_type in module_map:
        if isinstance(module_linear, types):
            batchnorm_type = target_type
            break
    if batchnorm_type is None:
        raise RuntimeError('No batchnorm type for linear layer found.')

    feature_index = None
    for types, index in feature_index_map:
        if isinstance(module_linear, types):
            feature_index = index
            break
    if feature_index is None:
        raise RuntimeError('No feature index for linear layer found.')

    return batchnorm_type(num_features=module_linear.weight.shape[feature_index]).eval()


@pytest.fixture(scope='session')
def data_input(rng, module_linear):
    shape = (16,)
    setups = [
        (Conv1d, 1, 1),
        (ConvTranspose1d, 0, 1),
        (Conv2d, 1, 2),
        (ConvTranspose2d, 0, 2),
        (Conv3d, 1, 3),
        (ConvTranspose3d, 0, 3)
    ]
    if isinstance(module_linear, Linear):
        shape += (module_linear.weight.shape[1],)
    else:
        for module_type, dim, ndims in setups:
            if isinstance(module_linear, module_type):
                shape += (module_linear.weight.shape[dim],) + (16,) * ndims

    return torch.empty(*shape).normal_(generator=rng)
