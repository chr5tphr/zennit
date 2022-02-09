'''Configuration and fixtures for testing'''
from itertools import product

import pytest
import torch
from torch.nn import Conv1d, ConvTranspose1d, Linear
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import Conv3d, ConvTranspose3d
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


def pytest_addoption(parser):
    '''Add options to pytest.'''
    parser.addoption(
        '--batchsize',
        default=4,
        help='Batch-size for generated samples.'
    )


def pytest_generate_tests(metafunc):
    '''Generate test fixture values based on CLI options.'''
    if 'batchsize' in metafunc.fixturenames:
        metafunc.parametrize('batchsize', [metafunc.config.getoption('batchsize')], scope='session')


def prodict(**kwargs):
    '''Create a dictionary with values which are the cartesian product of the input keyword arguments.'''
    return [dict(zip(kwargs, val)) for val in product(*kwargs.values())]


@pytest.fixture(
    scope='session',
    params=[
        0xdeadbeef,
        0xd0c0ffee,
        *[pytest.param(seed, marks=pytest.mark.extended) for seed in [
            0xc001bee5, 0xc01dfee7, 0xbe577001, 0xca7b0075, 0x1057b0a7, 0x900ddeed
        ]],
    ],
    ids=hex
)
def rng(request):
    '''Random number generator fixture.'''
    return torch.manual_seed(request.param)


@pytest.fixture(
    scope='session',
    params=[
        (torch.nn.ReLU, {}),
        (torch.nn.Softmax, dict(dim=1)),
        (torch.nn.Tanh, {}),
        (torch.nn.Sigmoid, {}),
        (torch.nn.Softplus, dict(beta=1)),
    ],
    ids=lambda param: param[0].__name__
)
def module_simple(rng, request):
    '''Fixture for simple modules.'''
    module_type, kwargs = request.param
    return module_type(**kwargs).to(torch.float64).eval()


@pytest.fixture(
    scope='session',
    params=[
        *product(
            [Linear],
            prodict(in_features=[16], out_features=[15], bias=[True, False]),
        ),
        *product(
            [Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d],
            prodict(in_channels=[1, 3], out_channels=[1, 3], kernel_size=[2, 3], bias=[True, False]),
        ),
    ],
    ids=lambda param: param[0].__name__
)
def module_linear(rng, request):
    '''Fixture for linear modules.'''
    module_type, kwargs = request.param
    return module_type(**kwargs).to(torch.float64).eval()


@pytest.fixture(scope='session')
def module_batchnorm(module_linear):
    '''Fixture for BatchNorm-type modules, based on adjacent linear module.'''
    module_map = [
        ((Linear, Conv1d, ConvTranspose1d), BatchNorm1d),
        ((Conv2d, ConvTranspose2d), BatchNorm2d),
        ((Conv3d, ConvTranspose3d), BatchNorm3d),
    ]
    feature_index_map = [
        ((ConvTranspose1d, ConvTranspose2d, ConvTranspose3d), 1),
        ((Linear, Conv1d, Conv2d, Conv3d), 0),
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

    return batchnorm_type(num_features=module_linear.weight.shape[feature_index]).to(torch.float64).eval()


@pytest.fixture(scope='session')
def data_linear(rng, batchsize, module_linear):
    '''Fixture to create data for a linear module, given an RNG.'''
    shape = (batchsize,)
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
                shape += (module_linear.weight.shape[dim],) + (4,) * ndims

    return torch.empty(*shape, dtype=torch.float64).normal_(generator=rng)


@pytest.fixture(scope='session', params=[
    (16,),
    (4,),
    (4, 4),
    (4, 4, 4),
])
def data_simple(request, rng, batchsize):
    '''Fixture to create data for a linear module, given an RNG.'''
    shape = (batchsize,) + request.param
    return torch.empty(*shape, dtype=torch.float64).normal_(generator=rng)
