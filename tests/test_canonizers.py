'''Tests for canonizers'''
import torch

from zennit.canonizers import SequentialMergeBatchNorm


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
