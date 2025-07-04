'''Helper functions for various tests.'''
from itertools import product

import numpy as np
import torch

from zennit.types import BatchNorm


def prodict(**kwargs):
    '''Create a dictionary with values which are the cartesian product of the input keyword arguments.'''
    return [dict(zip(kwargs, val)) for val in product(*kwargs.values())]


def one_hot_max(output):
    '''Get the one-hot encoded max.'''
    return torch.sparse_coo_tensor(
        [*zip(np.unravel_index(output.argmax(), output.shape))], [1.], output.shape, dtype=output.dtype
    ).to_dense()


def assert_identity_hook(equal=True, message=''):
    '''Create an assertion hook which checks whether the module does or does not modify its input.'''
    def assert_identity(module, input, output):
        '''Assert whether the module does or does not modify its input.'''
        assert equal == torch.allclose(input[0], output, rtol=1e-5), message
    return assert_identity


def randomize_bnorm(model):
    '''Randomize all BatchNorm module parameters of a model.'''
    for module in model.modules():
        if isinstance(module, BatchNorm):
            module.weight.data.uniform_(0.1, 2.0)
            module.running_var.data.uniform_(0.1, 2.0)
            module.bias.data.normal_()
            module.running_mean.data.normal_()
            # smaller eps to reduce error
            module.eps = 1e-30
    return model


def nograd(model):
    '''Unset grad requirement for all model parameters.'''
    for param in model.parameters():
        param.requires_grad = False
    return model
