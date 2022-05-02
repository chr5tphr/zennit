'''Tests for canonizers'''
from collections import OrderedDict
from functools import partial

import pytest
import torch
from torch.nn import Sequential
from helpers import assert_identity_hook

from zennit.canonizers import Canonizer, CompositeCanonizer
from zennit.canonizers import SequentialMergeBatchNorm, NamedMergeBatchNorm, AttributeCanonizer
from zennit.core import RemovableHandleList
from zennit.types import BatchNorm


def test_merge_batchnorm_consistency(module_linear, module_batchnorm, data_linear):
    '''Test whether the output of the merged batchnorm is consistent with its original output.'''
    output_linear_before = module_linear(data_linear)
    output_batchnorm_before = module_batchnorm(output_linear_before)
    canonizer = SequentialMergeBatchNorm()

    try:
        canonizer.register((module_linear,), module_batchnorm)
        output_linear_canonizer = module_linear(data_linear)
        output_batchnorm_canonizer = module_batchnorm(output_linear_canonizer)
    finally:
        canonizer.remove()

    output_linear_after = module_linear(data_linear)
    output_batchnorm_after = module_batchnorm(output_linear_after)

    assert all(torch.allclose(left, right, atol=1e-5) for left, right in [
        (output_linear_before, output_linear_after),
        (output_batchnorm_before, output_batchnorm_after),
        (output_batchnorm_before, output_linear_canonizer),
        (output_linear_canonizer, output_batchnorm_canonizer),
    ])


@pytest.mark.parametrize('canonizer_fn', [
    SequentialMergeBatchNorm,
    partial(NamedMergeBatchNorm, [(['dense0'], 'bnorm0')]),
])
def test_merge_batchnorm_apply(canonizer_fn, module_linear, module_batchnorm, data_linear):
    '''Test whether SequentialMergeBatchNorm merges BatchNorm modules correctly and keeps the output unchanged.'''
    model = Sequential(OrderedDict([
        ('dense0', module_linear),
        ('bnorm0', module_batchnorm)
    ]))
    output_before = model(data_linear)

    handles = RemovableHandleList(
        module.register_forward_hook(assert_identity_hook(True, 'BatchNorm was not merged!'))
        for module in model.modules() if isinstance(module, BatchNorm)
    )

    canonizer = canonizer_fn()

    canonizer_handles = RemovableHandleList(canonizer.apply(model))
    try:
        output_canonizer = model(data_linear)
    finally:
        handles.remove()
        canonizer_handles.remove()

    handles = RemovableHandleList(
        module.register_forward_hook(assert_identity_hook(False, 'BatchNorm was not restored!'))
        for module in model.modules() if isinstance(module, BatchNorm)
    )

    try:
        output_after = model(data_linear)
    finally:
        handles.remove()

    assert torch.allclose(output_canonizer, output_before, rtol=1e-5), 'Canonizer changed output after register!'
    assert torch.allclose(output_before, output_after, rtol=1e-5), 'Canonizer changed output after remove!'


def test_attribute_canonizer(module_linear, data_linear):
    '''Test whether AttributeCanonizer overwrites and restores a linear module's forward correctly. '''
    model = Sequential(OrderedDict([
        ('dense0', module_linear),
    ]))
    output_before = model(data_linear)

    modules = [module_linear]
    module_type = type(module_linear)

    assert all(
        module.forward == module_type.forward.__get__(module) for module in modules
    ), 'Model has its forward already overwritten!'

    def attribute_map(name, module):
        if module is module_linear:
            return {'forward': lambda x: module_type.forward.__get__(module)(x) * 2}
        return None

    canonizer = AttributeCanonizer(attribute_map)

    handles = RemovableHandleList(canonizer.apply(model))
    try:
        assert not any(
            module.forward == module_type.forward.__get__(module) for module in modules
        ), 'Model forward was not overwritten!'
        output_canonizer = model(data_linear)
    finally:
        handles.remove()

    output_after = model(data_linear)

    assert all(
        module.forward == module_type.forward.__get__(module) for module in modules
    ), 'Model forward was not restored!'
    assert torch.allclose(output_canonizer, output_before * 2, rtol=1e-5), 'Canonizer output mismatch after register!'
    assert torch.allclose(output_before, output_after, rtol=1e-5), 'Canonizer changed output after remove!'


def test_composite_canonizer():
    '''Test whether CompositeCanonizer correctly combines two AttributCanonizer canonizers.'''
    module_vanilla = torch.nn.Module()
    model = torch.nn.Sequential(module_vanilla)

    canonizer = CompositeCanonizer([
        AttributeCanonizer(lambda name, module: {'_test_x': 13}),
        AttributeCanonizer(lambda name, module: {'_test_y': 13}),
    ])

    handles = RemovableHandleList(canonizer.apply(model))
    try:
        assert hasattr(module_vanilla, '_test_x'), 'Model attribute _test_x was not overwritten!'
        assert hasattr(module_vanilla, '_test_y'), 'Model attribute _test_y was not overwritten!'
    finally:
        handles.remove()

    assert not hasattr(module_vanilla, '_test_x'), 'Model attribute _test_x was not removed!'
    assert not hasattr(module_vanilla, '_test_y'), 'Model attribute _test_y was not removed!'


def test_base_canonizer_cooperative_apply():
    '''Test wheter Canonizer's apply method is cooperative.'''

    class DummyCanonizer(Canonizer):
        '''Class to test Canonizer's cooperative apply.'''
        def apply(self, root_module):
            '''Cooperative apply which appends a string 'dummy' to the result of the parent class.'''
            instances = super().apply(root_module)
            instances += ['dummy']
            return instances

        def register(self):
            '''No-op register for abstract method.'''

        def remove(self):
            '''No-op remove for abstract method.'''

    canonizer = DummyCanonizer()
    model = Sequential()
    instances = canonizer.apply(model)
    assert 'dummy' in instances, 'Unexpected canonizer instance list!'
