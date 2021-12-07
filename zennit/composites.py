# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/composites.py
#
# Zennit is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Zennit is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
'''Composites, registered in a global composite dict.'''
import torch

from .core import Composite
from .layer import Sum
from .rules import Gamma, Epsilon, ZBox, ZPlus, AlphaBeta, Flat, Pass, Norm, ReLUDeconvNet, ReLUGuidedBackprop
from .types import Convolution, Linear, AvgPool, Activation


class LayerMapComposite(Composite):
    '''A Composite for which hooks are specified by a mapping from module types to hooks.

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


class SpecialFirstLayerMapComposite(LayerMapComposite):
    '''A Composite for which hooks are specified by a mapping from module types to hooks.

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
        occurrence in another mapping.

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


class NameMapComposite(Composite):
    '''A Composite for which hooks are specified by a mapping from module types to hooks.

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


COMPOSITES = {}


def register_composite(name):
    '''Register a composite in the global COMPOSITES dict under `name`.'''
    def wrapped(composite):
        '''Wrapped function to be called on the composite to register it to the global COMPOSITES dict.'''
        COMPOSITES[name] = composite
        return composite
    return wrapped


LAYER_MAP_BASE = [
    (Activation, Pass()),
    (Sum, Norm()),
    (AvgPool, Norm())
]


@register_composite('epsilon_gamma_box')
class EpsilonGammaBox(SpecialFirstLayerMapComposite):
    '''An explicit composite using the ZBox rule for the first convolutional layer, gamma rule for all following
    convolutional layers, and the epsilon rule for all fully connected layers.

    Parameters
    ----------
    low: obj:`torch.Tensor`
        A tensor with the same size as the input, describing the lowest possible pixel values.
    high: obj:`torch.Tensor`
        A tensor with the same size as the input, describing the highest possible pixel values.
    '''
    def __init__(self, low, high, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, Gamma(gamma=0.25)),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Convolution, ZBox(low, high))
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


@register_composite('epsilon_plus')
class EpsilonPlus(LayerMapComposite):
    '''An explicit composite using the zplus rule for all convolutional layers and the epsilon rule for all fully
    connected layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, ZPlus()),
            (torch.nn.Linear, Epsilon()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('epsilon_alpha2_beta1')
class EpsilonAlpha2Beta1(LayerMapComposite):
    '''An explicit composite using the alpha2-beta1 rule for all convolutional layers and the epsilon rule for all
    fully connected layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, AlphaBeta(alpha=2, beta=1)),
            (torch.nn.Linear, Epsilon()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('epsilon_plus_flat')
class EpsilonPlusFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, ZPlus()),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


@register_composite('epsilon_alpha2_beta1_flat')
class EpsilonAlpha2Beta1Flat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the alpha2-beta1 rule for all other
    convolutional layers and the epsilon rule for all other fully connected layers.
    '''
    def __init__(self, canonizers=None):
        layer_map = LAYER_MAP_BASE + [
            (Convolution, AlphaBeta(alpha=2, beta=1)),
            (torch.nn.Linear, Epsilon()),
        ]
        first_map = [
            (Linear, Flat())
        ]
        super().__init__(layer_map, first_map, canonizers=canonizers)


@register_composite('deconvnet')
class DeconvNet(LayerMapComposite):
    '''An explicit composite modifying the gradients of all ReLUs according to DeconvNet [1]_.

    References
    ----------
    .. [1] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in European conference
           on computer vision. Springer, 2014, pp. 818–833.
    '''
    def __init__(self, canonizers=None):
        layer_map = [
            (torch.nn.ReLU, ReLUDeconvNet()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('guided_backprop')
class GuidedBackprop(LayerMapComposite):
    '''An explicit composite modifying the gradients of all ReLUs according to GuidedBackprop [2]_.

    References
    ----------
    .. [2] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. A. Riedmiller, “Striving for simplicity: The all
           convolutional net,” in Proceedings of the International Conference of Learning Representations (ICLR), 2015.
    '''
    def __init__(self, canonizers=None):
        layer_map = [
            (torch.nn.ReLU, ReLUGuidedBackprop()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('excitation_backprop')
class ExcitationBackprop(LayerMapComposite):
    '''An explicit composite implementing the ExcitationBackprop [3]_.

    References
    ----------
    .. [3] J. Zhang, S. A. Bargal, Z. Lin, J. Brandt, X. Shen, and S. Sclaroff, “Top-down neural attention by
           excitation backprop,” International Journal of Computer Vision, vol. 126, no. 10, pp. 1084–1102, 2018.

    '''
    def __init__(self, canonizers=None):
        layer_map = [
            (Sum, Norm()),
            (AvgPool, Norm()),
            (Linear, ZPlus()),
        ]
        super().__init__(layer_map, canonizers=canonizers)
