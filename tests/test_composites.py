'''Tests for composites using torchvision models.'''
from types import MethodType
from itertools import product

import pytest
from torchvision.models import vgg11, resnet18

from zennit.core import BasicHook, collect_leaves
from zennit.composites import COMPOSITES, NameMapComposite, LayerMapComposite, SpecialFirstLayerMapComposite
from zennit.composites import EpsilonGammaBox


@pytest.fixture(scope='session', params=[vgg11, resnet18])
def composite_model(request):
    '''Models to test composites on.'''
    return request.param()


def ishookcopy(hook, hook_template):
    '''Check if ``hook`` is a copy of ``hook_template`` (due to copying-mechanics of BasicHook).'''
    if isinstance(hook_template, BasicHook):
        return all(
            getattr(hook, key) == getattr(hook_template, key)
            for key in (
                'input_modifiers',
                'param_modifiers',
                'output_modifiers',
                'gradient_mapper',
                'reducer',
                'param_keys',
                'require_params'
            )
        )
    return isinstance(hook, type(hook_template))


def check_hook_registered(module, hook_template):
    '''Check whether a ``hook_template`` has been registered to ``module``. '''
    return any(
        ishookcopy(hook_func.__self__, hook_template)
        for hook_func in getattr(module, '_forward_pre_hooks').values()
        if isinstance(hook_func, MethodType)
    )


def verify_no_hooks(model):
    '''Verify that ``model`` has no registered forward (-pre) hooks.'''
    return not any(
        any(getattr(module, key) for key in ('_forward_hooks', '_forward_pre_hooks'))
        for module in model.modules()
    )


SPECIAL_FIRST_LAYER_MAP_COMPOSITES = [
    elem for elem in COMPOSITES.values() if issubclass(elem, SpecialFirstLayerMapComposite)
]
LAYER_MAP_COMPOSITES = [
    elem for elem in COMPOSITES.values()
    if issubclass(elem, LayerMapComposite) and not issubclass(elem, SpecialFirstLayerMapComposite)
]
COMPOSITE_KWARGS = {
    EpsilonGammaBox: {'low': -3., 'high': 3.},
}


@pytest.fixture(scope='session', params=LAYER_MAP_COMPOSITES)
def layer_map_composite(request):
    '''Fixture for explicit LayerMapComposites.'''
    return request.param(**COMPOSITE_KWARGS.get(request.param, {}))


@pytest.fixture(scope='session', params=SPECIAL_FIRST_LAYER_MAP_COMPOSITES)
def special_first_layer_map_composite(request):
    '''Fixturer for explicit SpecialFirstLayerMapComposites.'''
    return request.param(**COMPOSITE_KWARGS.get(request.param, {}))


@pytest.fixture(scope='session')
def name_map_composite(request, composite_model, layer_map_composite):
    '''Fixture to create NameMapComposites based on explicit LayerMapComposites.'''
    rule_map = {}
    for name, child in composite_model.named_modules():
        for dtype, hook_template in layer_map_composite.layer_map:
            if isinstance(child, dtype):
                rule_map.setdefault(hook_template, []).append(name)
                break
    name_map = [(tuple(value), key) for key, value in rule_map.items()]
    return NameMapComposite(name_map=name_map)


def test_composite_layer_map_registered(layer_map_composite, composite_model):
    '''Tests whether the explicit LayerMapComposites register and unregister their rules correctly.'''
    errors = []
    with layer_map_composite.context(composite_model):
        for child, (dtype, hook_template) in product(composite_model.modules(), layer_map_composite.layer_map):
            if isinstance(child, dtype) and not check_hook_registered(child, hook_template):
                errors.append((
                    '{} is of {} but {} is not registered!',
                    (child, dtype, hook_template),
                ))

    if not verify_no_hooks(composite_model):
        errors.append(('Model has hooks registered after composite was removed!', ()))

    assert not errors, 'Errors:\n  ' + '\n  '.join(f'[{n}] ' + msg.format(*arg) for n, (msg, arg) in enumerate(errors))


def test_composite_special_first_layer_map_registered(special_first_layer_map_composite, composite_model):
    '''Tests whether the explicit LayerMapComposites register and unregister their rules correctly.'''
    errors = []
    try:
        special_first_layer, special_first_template, special_first_dtype = next(
            (child, hook_template, dtype)
            for child, (dtype, hook_template) in product(
                collect_leaves(composite_model), special_first_layer_map_composite.first_map
            ) if isinstance(child, dtype)
        )
    except StopIteration:
        special_first_layer = None
        special_first_template = None

    with special_first_layer_map_composite.context(composite_model):
        if special_first_layer is not None and not check_hook_registered(special_first_layer, special_first_template):
            errors.append((
                'Special first layer {} is of {} but {} is not registered!',
                (special_first_layer, special_first_dtype, special_first_template)
            ))

        children = (child for child in composite_model.modules() if child is not special_first_layer)
        for child, (dtype, hook_template) in product(children, special_first_layer_map_composite.layer_map):
            if isinstance(child, dtype) and not check_hook_registered(child, hook_template):
                errors.append((
                    '{} is of {} but {} is not registered!',
                    (child, dtype, hook_template),
                ))

    if not verify_no_hooks(composite_model):
        errors.append(('Model has hooks registered after composite was removed!', ()))

    assert not errors, 'Errors:\n  ' + '\n  '.join(f'[{n}] ' + msg.format(*arg) for n, (msg, arg) in enumerate(errors))


def test_composite_name_map_registered(name_map_composite, composite_model):
    '''Tests whether the constructed NameMapComposites register and unregister their rules correctly.'''
    errors = []
    setups = product(composite_model.named_modules(), name_map_composite.name_map)
    with name_map_composite.context(composite_model):
        for (name, child), (names, hook_template) in setups:
            if name in names and not check_hook_registered(child, hook_template):
                errors.append(
                    '{} is in name map for {}, but is not registered!',
                    (name, hook_template),
                )

    if not verify_no_hooks(composite_model):
        errors.append(('Model has hooks registered after composite was removed!', ()))

    assert not errors, 'Errors:\n  ' + '\n  '.join(f'[{n}] ' + msg.format(*arg) for n, (msg, arg) in enumerate(errors))
