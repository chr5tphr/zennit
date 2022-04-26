'''Tests for torchvision-model-specific canonizers.'''
import pytest
import torch
from torchvision.models import vgg11_bn, resnet18, resnet50
from torchvision.models.resnet import BasicBlock as ResNetBasicBlock, Bottleneck as ResNetBottleneck

from zennit.core import Composite, RemovableHandleList
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.types import BatchNorm


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


def test_vgg_canonizer(batchsize):
    '''Test whether VGGCanonizer merges BatchNorm modules correctly and keeps the output unchanged.'''
    model = randomize_bnorm(nograd(vgg11_bn().eval().to(torch.float64)))
    data = torch.randn((batchsize, 3, 224, 224), dtype=torch.float64)
    output_before = model(data)

    handles = RemovableHandleList(
        module.register_forward_hook(assert_identity_hook(True, 'BatchNorm was not merged!'))
        for module in model.modules() if isinstance(module, BatchNorm)
    )

    canonizer = VGGCanonizer()
    composite = Composite(canonizers=[canonizer])

    try:
        composite.register(model)
        output_canonizer = model(data)
    finally:
        composite.remove()
        handles.remove()

    # this assumes the batch-norm is not initialized as the identity
    handles = RemovableHandleList(
        module.register_forward_hook(assert_identity_hook(False, 'BatchNorm was not restored!'))
        for module in model.modules() if isinstance(module, BatchNorm)
    )
    try:
        output_after = model(data)
    finally:
        handles.remove()

    assert torch.allclose(output_canonizer, output_before, rtol=1e-5), 'Canonizer changed output after register!'
    assert torch.allclose(output_before, output_after, rtol=1e-5), 'Canonizer changed output after remove!'


@pytest.mark.parametrize('model_fn,block_type', [
    (resnet18, ResNetBasicBlock),
    (resnet50, ResNetBottleneck),
])
def test_resnet_canonizer(batchsize, model_fn, block_type):
    '''Test whether ResNetCanonizer overwrites and restores the Bottleneck/BasicBlock forward, merges BatchNorm modules
    correctly and keeps the output unchanged.
    '''
    model = randomize_bnorm(nograd(model_fn().eval().to(torch.float64)))
    data = torch.randn((batchsize, 3, 224, 224), dtype=torch.float64)
    blocks = [module for module in model.modules() if isinstance(module, block_type)]

    assert blocks, 'Model has no blocks!'
    assert all(
        block.forward == block_type.forward.__get__(block) for block in blocks
    ), 'Model has its forward already overwritten!'

    output_before = model(data)

    handles = RemovableHandleList(
        module.register_forward_hook(assert_identity_hook(True, 'BatchNorm was not merged!'))
        for module in model.modules() if isinstance(module, BatchNorm)
    )

    canonizer = ResNetCanonizer()
    composite = Composite(canonizers=[canonizer])

    try:
        composite.register(model)
        assert not any(
            block.forward == block_type.forward.__get__(block) for block in blocks
        ), 'Model forward was not overwritten!'
        output_canonizer = model(data)
    finally:
        composite.remove()
        handles.remove()

    # this assumes the batch-norm is not initialized as the identity
    handles = RemovableHandleList(
        module.register_forward_hook(assert_identity_hook(False, 'BatchNorm was not restored!'))
        for module in model.modules() if isinstance(module, BatchNorm)
    )
    try:
        output_after = model(data)
    finally:
        handles.remove()

    assert all(
        block.forward == block_type.forward.__get__(block) for block in blocks
    ), 'Model forward was not restored!'
    assert torch.allclose(output_canonizer, output_before, rtol=1e-5), 'Canonizer changed output after register!'
    assert torch.allclose(output_before, output_after, rtol=1e-5), 'Canonizer changed output after remove!'
