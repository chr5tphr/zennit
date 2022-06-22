'''Configuration and fixtures for testing'''
from itertools import product
from collections import OrderedDict

import pytest
import torch
from torch.nn import Conv1d, ConvTranspose1d, Linear
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import Conv3d, ConvTranspose3d
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torchvision.models import vgg11, resnet18, alexnet
from helpers import prodict, one_hot_max

from zennit.attribution import identity
from zennit.core import Composite, Hook
from zennit.composites import COMPOSITES, NameMapComposite, LayerMapComposite, SpecialFirstLayerMapComposite
from zennit.composites import EpsilonGammaBox
from zennit.types import Linear as AnyLinear, Activation


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
def module_batchnorm(module_linear, rng):
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

    batchnorm = batchnorm_type(num_features=module_linear.weight.shape[feature_index]).to(torch.float64).eval()
    batchnorm.weight.data.uniform_(**{'from': 0.1, 'to': 2.0, 'generator': rng})
    batchnorm.bias.data.normal_(generator=rng)
    batchnorm.eps = 1e-30
    return batchnorm


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


COMPOSITE_KWARGS = {
    EpsilonGammaBox: {'low': -3., 'high': 3.},
}


class PassClone(Hook):
    '''Clone of the Pass rule.'''
    def backward(self, module, grad_input, grad_output):
        '''Directly return grad_output.'''
        return grad_output


class GradClone(Hook):
    '''Explicit rule to return the cloned gradient.'''
    def backward(self, module, grad_input, grad_output):
        '''Directly return grad_output.'''
        return grad_input.clone()


@pytest.fixture(scope='session', params=[
    None,
    [(Linear, GradClone()), (Activation, PassClone())],
])
def cooperative_layer_map(request):
    '''Fixture for a cooperative layer map in LayerMapComposite subtypes.'''
    return request.param


@pytest.fixture(scope='session', params=[
    None,
    [(AnyLinear, GradClone())],
])
def cooperative_first_map(request):
    '''Fixture for a cooperative layer map for the first layer in SpecialFirstLayerMapComposite subtypes.'''
    return request.param


@pytest.fixture(scope='session', params=[
    elem for elem in COMPOSITES.values()
    if issubclass(elem, LayerMapComposite) and not issubclass(elem, SpecialFirstLayerMapComposite)
])
def layer_map_composite(request, cooperative_layer_map):
    '''Fixture for explicit LayerMapComposites.'''
    return request.param(layer_map=cooperative_layer_map, **COMPOSITE_KWARGS.get(request.param, {}))


@pytest.fixture(scope='session', params=[
    elem for elem in COMPOSITES.values() if issubclass(elem, SpecialFirstLayerMapComposite)
])
def special_first_layer_map_composite(request, cooperative_layer_map, cooperative_first_map):
    '''Fixturer for explicit SpecialFirstLayerMapComposites.'''
    return request.param(
        layer_map=cooperative_layer_map,
        first_map=cooperative_first_map,
        **COMPOSITE_KWARGS.get(request.param, {})
    )


@pytest.fixture(scope='session', params=[Composite, *COMPOSITES.values()])
def any_composite(request):
    '''Fixture for all explicitly registered Composites, as well as the empty Composite.'''
    return request.param(**COMPOSITE_KWARGS.get(request.param, {}))


@pytest.fixture(scope='session')
def name_map_composite(request, model_vision, layer_map_composite):
    '''Fixture to create NameMapComposites based on explicit LayerMapComposites.'''
    rule_map = {}
    for name, child in model_vision.named_modules():
        for dtype, hook_template in layer_map_composite.layer_map:
            if isinstance(child, dtype):
                rule_map.setdefault(hook_template, []).append(name)
                break
    name_map = [(tuple(value), key) for key, value in rule_map.items()]
    return NameMapComposite(name_map=name_map)


@pytest.fixture(scope='session', params=[alexnet, vgg11, resnet18])
def model_vision(request):
    '''Models to test composites on.'''
    return request.param()


@pytest.fixture(scope='session')
def model_simple(rng, module_linear, data_linear):
    '''Fixture for a simple model, using a linear module followed by a ReLU and a dense layer.'''
    with torch.no_grad():
        intermediate = module_linear(data_linear)
    return torch.nn.Sequential(OrderedDict([
        ('linr0', module_linear),
        ('actv0', torch.nn.ReLU()),
        ('flat0', torch.nn.Flatten()),
        ('linr1', torch.nn.Linear(intermediate.shape[1:].numel(), 4, dtype=intermediate.dtype)),
    ]))


@pytest.fixture(scope='session')
def model_simple_grad(data_linear, model_simple):
    '''Fixture for gradient wrt. data_linear for model_simple.'''
    data = data_linear.detach().requires_grad_()
    output = model_simple(data)
    grad, = torch.autograd.grad(output, data, output)
    return grad


@pytest.fixture(scope='session')
def model_simple_output(data_linear, model_simple):
    '''Fixture for output given data_linear for model_simple.'''
    data = data_linear.detach()
    output = model_simple(data)
    return output


@pytest.fixture(scope='session', params=[
    identity,
    one_hot_max,
    torch.ones_like,
])
def grad_outputs_func(request):
    '''Fixture for common attr_output_fn functions.'''
    return request.param
