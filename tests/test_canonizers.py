'''Tests for canonizers'''
from collections import OrderedDict
from functools import partial

import pytest
import torch
from torch.nn import Sequential
from helpers import assert_identity_hook

from zennit.canonizers import Canonizer, CompositeCanonizer, KMeansCanonizer
from zennit.canonizers import SequentialMergeBatchNorm, NamedMergeBatchNorm, AttributeCanonizer
from zennit.core import RemovableHandleList
from zennit.types import BatchNorm
from zennit.layer import PairwiseCentroidDistance


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
    '''Test whether CompositeCanonizer correctly combines two AttributeCanonizer canonizers.'''
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
    '''Test whether Canonizer's apply method is cooperative.'''

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


def test_kmeans_canonizer():
    '''Test whether KMeansCanonizer correctly modifies and restores a PairwiseCentroidDistance module.'''
    # Sample data for the KMeansCanonizer test, defined locally
    n_samples_kmeans = 10
    n_features_kmeans = 5
    n_clusters_kmeans = 3

    torch.manual_seed(0)  # For reproducibility
    sample_data = torch.randn(n_samples_kmeans, n_features_kmeans)
    centroids = torch.randn(n_clusters_kmeans, n_features_kmeans)

    # 1. Create the model with a PairwiseCentroidDistance layer
    # Important: power=2 is the condition for KMeansCanonizer to apply
    original_distance_layer = PairwiseCentroidDistance(centroids.clone(), power=2)
    model = Sequential(OrderedDict([
        ('distance', original_distance_layer)
    ]))
    model.eval()  # Set to evaluation mode

    # 2. Output of the model *before* canonization
    output_before = model(sample_data)
    assignments_before = torch.argmin(output_before, dim=1)

    # 3. Apply the KMeansCanonizer
    canonizer = KMeansCanonizer()
    applied_instances = canonizer.apply(model)
    handles = RemovableHandleList(applied_instances)

    assert isinstance(model.distance, torch.nn.Sequential), \
        "PairwiseCentroidDistance module was not replaced by a Sequential module."
    assert not isinstance(model.distance, PairwiseCentroidDistance), \
        "Type check of PairwiseCentroidDistance module failed after replacement."
    assert len(model.distance) == 3, \
        f"Canonized module should be a Sequential of 3 modules, got {len(model.distance)}"
    assert isinstance(model.distance[0], torch.nn.Module), \
        "First module in canonized sequence is not a torch.nn.Module (expected NeuralizedKMeans)."
    assert isinstance(model.distance[1], torch.nn.Module), \
        "Second module in canonized sequence is not a torch.nn.Module (expected MinPool1d)."
    assert isinstance(model.distance[2], torch.nn.Flatten), \
        "Third module in canonized sequence is not a torch.nn.Flatten."

    try:
        # 4. Output of the model *after* canonization
        output_canonized = model(sample_data)
        assignments_canonized = torch.argmax(output_canonized, dim=1)

        # 5. Verify that cluster assignments match
        assert torch.equal(assignments_canonized, assignments_before), (
            f"Cluster assignments differ after canonization.\n"
            f"Original assignments: {assignments_before}\n"
            f"Canonized assignments: {assignments_canonized}"
        )

        for i in range(sample_data.shape[0]):
            assigned_idx = assignments_before[i].item()
            for k_cluster in range(centroids.shape[0]):
                val = output_canonized[i, k_cluster].item()
                if k_cluster == assigned_idx:
                    assert val >= -1e-5, (
                        f"Sample {i}, assigned cluster {k_cluster}: Output {val} "
                        f"should be >= -1e-5. All outputs: {output_canonized[i]}"
                    )
                else:
                    assert val < 1e-5, (
                        f"Sample {i}, non-assigned cluster {k_cluster}: Output {val} "
                        f"should be < 1e-5. All outputs: {output_canonized[i]}"
                    )
    finally:
        # 6. Remove the canonizer (restore original state)
        handles.remove()

    assert isinstance(model.distance, PairwiseCentroidDistance), \
        "PairwiseCentroidDistance module was not restored."
    assert model.distance is original_distance_layer, \
        "The original instance of the PairwiseCentroidDistance module was not restored."

    # 7. Output of the model *after* removing the canonizer
    output_after = model(sample_data)
    assert torch.allclose(output_after, output_before, atol=1e-6), \
        "Output changed after removing the canonizer.\n" \
        f"Output before: {output_before}\n" \
        f"Output after: {output_after}"
