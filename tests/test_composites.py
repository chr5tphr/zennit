'''Tests for composites using torchvision models.'''
from types import MethodType
from itertools import product

from zennit.core import BasicHook, collect_leaves


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


def test_composite_layer_map_registered(layer_map_composite, model_vision):
    '''Tests whether the explicit LayerMapComposites register and unregister their rules correctly.'''
    errors = []
    with layer_map_composite.context(model_vision):
        for child in model_vision.modules():
            for dtype, hook_template in layer_map_composite.layer_map:
                if isinstance(child, dtype):
                    if not check_hook_registered(child, hook_template):
                        errors.append((
                            '{} is first of {} but {} is not registered!',
                            (child, dtype, hook_template),
                        ))
                    break

    if not verify_no_hooks(model_vision):
        errors.append(('Model has hooks registered after composite was removed!', ()))

    assert not errors, 'Errors:\n  ' + '\n  '.join(f'[{n}] ' + msg.format(*arg) for n, (msg, arg) in enumerate(errors))


def test_composite_special_first_layer_map_registered(special_first_layer_map_composite, model_vision):
    '''Tests whether the explicit LayerMapComposites register and unregister their rules correctly.'''
    errors = []
    try:
        special_first_layer, special_first_template, special_first_dtype = next(
            (child, hook_template, dtype)
            for child, (dtype, hook_template) in product(
                collect_leaves(model_vision), special_first_layer_map_composite.first_map
            ) if isinstance(child, dtype)
        )
    except StopIteration:
        special_first_layer = None
        special_first_template = None

    with special_first_layer_map_composite.context(model_vision):
        if special_first_layer is not None and not check_hook_registered(special_first_layer, special_first_template):
            errors.append((
                'Special first layer {} is first of {} but {} is not registered!',
                (special_first_layer, special_first_dtype, special_first_template)
            ))

        children = (child for child in model_vision.modules() if child is not special_first_layer)
        for child in children:
            for dtype, hook_template in special_first_layer_map_composite.layer_map:
                if isinstance(child, dtype):
                    if not check_hook_registered(child, hook_template):
                        errors.append((
                            '{} is first of {} but {} is not registered!',
                            (child, dtype, hook_template),
                        ))
                    break

    if not verify_no_hooks(model_vision):
        errors.append(('Model has hooks registered after composite was removed!', ()))

    assert not errors, 'Errors:\n  ' + '\n  '.join(f'[{n}] ' + msg.format(*arg) for n, (msg, arg) in enumerate(errors))


def test_composite_name_map_registered(name_map_composite, model_vision):
    '''Tests whether the constructed NameMapComposites register and unregister their rules correctly.'''
    errors = []
    with name_map_composite.context(model_vision):
        for name, child in model_vision.named_modules():
            for names, hook_template in name_map_composite.name_map:
                if name in names:
                    if not check_hook_registered(child, hook_template):
                        errors.append(
                            '{} is first in name map for {}, but is not registered!',
                            (name, hook_template),
                        )
                    break

    if not verify_no_hooks(model_vision):
        errors.append(('Model has hooks registered after composite was removed!', ()))

    assert not errors, 'Errors:\n  ' + '\n  '.join(f'[{n}] ' + msg.format(*arg) for n, (msg, arg) in enumerate(errors))
