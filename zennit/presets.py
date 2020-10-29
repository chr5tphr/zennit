'''Presets, registered in a global preset dict.'''
import torch

from .canonizers import MergeBatchNorm
from .core import Preset
from .rules import Gamma, Epsilon, ZBox
from .types import Convolution


PRESETS = {}


def register_preset(name):
    '''Register a preset in the global PRESETS dict under `name`.'''
    def wrapped(preset):
        '''Wrapped function to be called on the preset to register it to the global PRESETS dict.'''
        PRESETS[name] = preset
        return preset
    return wrapped


class LayerMapPreset(Preset):
    '''A Preset for which hooks are specified by a mapping from module types to hooks.

    Parameters
    ----------
    layer_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook.
    '''
    def __init__(self, layer_map, canonizers=None):
        self.layer_map = layer_map
        super().__init__(self.mapping, canonizers)

    # pylint: disable=unused-argument
    def mapping(self, ctx, name, module):
        '''Get the appropriate hook given a mapping from module types to hooks.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found with the module type in the given layer map, or None if no applicable hook was found.
        '''
        return next((hook for types, hook in self.layer_map if isinstance(module, types)), None)


class SpecialFirstLayerMapPreset(LayerMapPreset):
    '''A Preset for which hooks are specified by a mapping from module types to hooks.

    Parameters
    ----------
    layer_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`.
    '''
    def __init__(self, layer_map, first_map, canonizers=None):
        self.first_map = first_map
        super().__init__(layer_map, canonizers)

    def mapping(self, ctx, name, module):
        '''Get the appropriate hook given a mapping from module types to hooks and a special mapping for the first
        occurence in another mapping.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found with the module type in the given layer map, in the first layer map if this was the first
            layer, or None if no applicable hook was found.
        '''
        if not ctx.get('first_layer_visited', False):
            for types, hook in self.first_map:
                if isinstance(module, types):
                    ctx['first_layer_visited'] = True
                    return hook

        return super().mapping(ctx, name, module)


class NameMapPreset(Preset):
    '''A Preset for which hooks are specified by a mapping from module types to hooks.

    Parameters
    ----------
    name_map: `list[tuple[tuple[str, ...], Hook]]`
        A mapping as a list of tuples, with a tuple of applicable module names and a Hook.
    '''
    def __init__(self, name_map, canonizers=None):
        self.name_map = name_map
        super().__init__(self.mapping, canonizers)

    # pylint: disable=unused-argument
    def mapping(self, ctx, name, module):
        '''Get the appropriate hook given a mapping from module names to hooks.

        Parameters
        ----------
        ctx: dict
            A context dictionary to keep track of previously registered hooks.
        name: str
            Name of the module.
        module: obj:`torch.nn.Module`
            Instance of the module to find a hook for.

        Returns
        -------
        obj:`Hook` or None
            The hook found with the module type in the given name map, or None if no applicable hook was found.
        '''
        return next((hook for names, hook in self.name_map if name in names), None)


register_preset('gamma_epsilon')
class GammaEpsilon(SpecialFirstLayerMapPreset):
    '''An explicit preset using the ZBox rule for the first convolutional layer, gamma rule for all following
    convolutional layers, and the epsilon rule for all fully connected layers.

    Parameters
    ----------
    low: obj:`torch.Tensor`
        A tensor with the same size as the input, describing the lowest possible pixel values.
    high: obj:`torch.Tensor`
        A tensor with the same size as the input, describing the highest possible pixel values.
    '''
    def __init__(self, low, high):
        layer_map = [
            (Convolution, Gamma(gamma=0.25)),
            (torch.nn.Linear, Epsilon())
        ]
        first_map = [
            (Convolution, ZBox(low, high))
        ]
        super().__init__(layer_map, first_map, canonizers=[MergeBatchNorm])
