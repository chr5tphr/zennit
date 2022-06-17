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
from .types import Convolution, Linear, AvgPool, Activation, BatchNorm


class LayerMapComposite(Composite):
    '''A Composite for which hooks are specified by a mapping from module types to hooks.

    Parameters
    ----------
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, layer_map, canonizers=None):
        self.layer_map = layer_map
        super().__init__(module_map=self.mapping, canonizers=canonizers)

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
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, layer_map, first_map, canonizers=None):
        self.first_map = first_map
        super().__init__(layer_map=layer_map, canonizers=canonizers)

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
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, name_map, canonizers=None):
        self.name_map = name_map
        super().__init__(module_map=self.mapping, canonizers=canonizers)

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


def layer_map_base(stabilizer=1e-6):
    '''Return a basic layer map (list of 2-tuples) shared by all built-in LayerMapComposites.

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.

    Returns
    -------
    list[tuple[tuple[torch.nn.Module, ...], Hook]]
        Basic ayer map shared by all built-in LayerMapComposites.
    '''
    return [
        (Activation, Pass()),
        (Sum, Norm(stabilizer=stabilizer)),
        (AvgPool, Norm(stabilizer=stabilizer)),
        (BatchNorm, Pass()),
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
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    gamma: float, optional
        Gamma parameter for the gamma rule.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`. This will be prepended to the ``first_map``
        defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(
        self,
        low,
        high,
        epsilon=1e-6,
        gamma=0.25,
        stabilizer=1e-6,
        layer_map=None,
        first_map=None,
        zero_params=None,
        canonizers=None
    ):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, Gamma(gamma=gamma, stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Convolution, ZBox(low=low, high=high, stabilizer=stabilizer, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)


@register_composite('epsilon_plus')
class EpsilonPlus(LayerMapComposite):
    '''An explicit composite using the zplus rule for all convolutional layers and the epsilon rule for all fully
    connected layers.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, epsilon=1e-6, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, ZPlus(stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


@register_composite('epsilon_alpha2_beta1')
class EpsilonAlpha2Beta1(LayerMapComposite):
    '''An explicit composite using the alpha2-beta1 rule for all convolutional layers and the epsilon rule for all
    fully connected layers.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, epsilon=1e-6, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, AlphaBeta(alpha=2, beta=1, stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


@register_composite('epsilon_plus_flat')
class EpsilonPlusFlat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the zplus rule for all other convolutional
    layers and the epsilon rule for all other fully connected layers.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`. This will be prepended to the ``first_map``
        defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(
        self, epsilon=1e-6, stabilizer=1e-6, layer_map=None, first_map=None, zero_params=None, canonizers=None
    ):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, ZPlus(stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Linear, Flat(stabilizer=stabilizer, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)


@register_composite('epsilon_alpha2_beta1_flat')
class EpsilonAlpha2Beta1Flat(SpecialFirstLayerMapComposite):
    '''An explicit composite using the flat rule for any linear first layer, the alpha2-beta1 rule for all other
    convolutional layers and the epsilon rule for all other fully connected layers.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`. This will be prepended to the ``first_map``
        defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(
        self, epsilon=1e-6, stabilizer=1e-6, layer_map=None, first_map=None, zero_params=None, canonizers=None
    ):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, AlphaBeta(alpha=2, beta=1, stabilizer=stabilizer, **rule_kwargs)),
            (torch.nn.Linear, Epsilon(epsilon=epsilon, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Linear, Flat(stabilizer=stabilizer, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)


@register_composite('deconvnet')
class DeconvNet(LayerMapComposite):
    '''An explicit composite modifying the gradients of all ReLUs according to DeconvNet
    :cite:p:`zeiler2014visualizing`.

    Parameters
    ----------
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, layer_map=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        layer_map = layer_map + [
            (torch.nn.ReLU, ReLUDeconvNet()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


@register_composite('guided_backprop')
class GuidedBackprop(LayerMapComposite):
    '''An explicit composite modifying the gradients of all ReLUs according to GuidedBackprop
    :cite:p:`springenberg2015striving`.

    Parameters
    ----------
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, layer_map=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        layer_map = layer_map + [
            (torch.nn.ReLU, ReLUGuidedBackprop()),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)


@register_composite('excitation_backprop')
class ExcitationBackprop(LayerMapComposite):
    '''An explicit composite implementing the ExcitationBackprop :cite:p:`zhang2016top`.

    Parameters
    ----------
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, stabilizer=1e-6, layer_map=None, zero_params=None, canonizers=None):
        if layer_map is None:
            layer_map = []

        layer_map = layer_map + [
            (Sum, Norm(stabilizer=stabilizer)),
            (AvgPool, Norm(stabilizer=stabilizer)),
            (Linear, ZPlus(stabilizer=stabilizer, zero_params=zero_params)),
        ]
        super().__init__(layer_map=layer_map, canonizers=canonizers)
