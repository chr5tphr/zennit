# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/attribution.py
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
'''Attributors are convienence objects to compute an attributions, optionally using composites.'''
from abc import ABCMeta, abstractmethod

import torch


def constant(obj):
    def wrapped_const(*args, **kwargs):
        return obj
    return wrapped_const


def identity(obj):
    return obj


class Attributor(metaclass=ABCMeta):
    '''Base Attributor Class.

    Attributors are convienience objects with an optional composite and when called, compute an attribution,
    e.g., the gradient or anything that is the result of computing the gradient when using the provided composite.
    Attributors also provide a context to be used in a `with` statement, similar to `CompositeContext`s. If the
    forward function (or self.__call__) is called and the composite has not been registered (i.e. `composite.handles`
    is empty), the composite will be temporarily registered to the model.

    Parameters
    ----------
    model: obj:`torch.nn.Module`
        The model for which the attribution will be computed. If `composite` is provided, this will also be the model
        to which the composite will be registered within `with` statements, or when calling the `Attributor` instance.
    composite: obj:`zennit.core.Composite`, optional
        The optional composite to, if provided, be registered to the model within `with` statements or when calling the
        `Attributor` instance.
    attr_output: obj:`torch.Tensor` or callable, optional
        The default output attribution to be used when calling the `Attributor` instance, which is either a Tensor
        compatible with any input used, or a function of the model's output. If None (default), the value will be the
        identity function.

    '''
    def __init__(self, model, composite=None, attr_output=None):
        self.model = model
        self.composite = composite

        if attr_output is None:
            self.attr_output_fn = identity
        elif not callable(attr_output):
            self.attr_output_fn = constant(attr_output)
        else:
            self.attr_output_fn = attr_output

    def __enter__(self):
        '''Register the composite, if provided.

        Returns
        -------
        self: obj:`Attributor`
            The `Attributor` instance.

        '''
        if self.composite is not None:
            self.composite.register(self.model)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''Remove the composite, if provided.

        Returns
        -------
        False

        '''
        if self.composite is not None:
            self.composite.remove()
        return False

    def __call__(self, input, attr_output=None):
        '''Compute the attribution of the model wrt. `input`, using `attr_output` as the output attribution if
        provided, or the default output attribution otherwise (if not supplied during instantiation either, this will
        be the full output of the model). If a composite was supplied to the `Attributor` instance, but it was not yet
        registered (either manually, or in a `with` statement), it will be registered to the model temporarily during
        the call.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution, which is either a Tensor compatible `input` (i.e. has the same shape as the output
            of the model), or a function of the model's output. If None (default), the default attribution will be
            used, which if neither supplied, will result in the model output used as the output attribution.

        Returns
        -------
        output: obj:`torch.Tensor`
            Output of the model with argument `input`.
        attribution: obj:`torch.Tensor`
            Attribution of the model wrt. to `input`, with the same shape as `input`.
        '''
        if attr_output is None:
            attr_output_fn = self.attr_output_fn
        elif not callable(attr_output):
            attr_output_fn = constant(attr_output)
        else:
            attr_output_fn = attr_output

        if self.composite is None or self.composite.handles:
            return self.forward(input, attr_output_fn)

        with self:
            return self.forward(input, attr_output_fn)

    @abstractmethod
    def forward(self, input, attr_output_fn):
        '''Abstract method. Compute the attribution of the model wrt. input, by using `attr_output_fn` as the function
        of the model output to provide the output attribution. This function will not register the composite, and is
        wrapped in the __call__ of `Attributor`.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution function of the model's output.
        '''


class Gradient(Attributor):
    '''The Gradient Attributor. The result is the product of the attribution output and the (possibly modified) jacobian.
    With a composite, i.e. `EpsilonGammaBox`, this will compute the Layerwise Relevance Propagation attribution values.
    '''
    def forward(self, input, attr_output_fn):
        '''Compute the gradient of the model wrt. input, by using `attr_output_fn` as the function of the model output
        to provide the vector for the vector jacobian product.
        This function will not register the composite, and is wrapped in the __call__ of `Attributor`.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution.
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution function of the model's output.

        Returns
        -------
        output: obj:`torch.Tensor`
            Output of the model given `input`.
        attribution: obj:`torch.Tensor`
            Attribution of the model wrt. to `input`, with the same shape as `input`.
        '''
        input = input.clone().requires_grad_(True)
        output = self.model(input)
        gradient, = torch.autograd.grad((output,), (input,), grad_outputs=(attr_output_fn(output.detach()),))
        return output, gradient
