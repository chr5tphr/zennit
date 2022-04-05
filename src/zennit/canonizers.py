# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/canonizers.py
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
'''Functions to produce a canonical form of models fit for LRP'''
from abc import ABCMeta, abstractmethod

import torch

from .core import collect_leaves
from .types import Linear, BatchNorm, ConvolutionTranspose


class Canonizer(metaclass=ABCMeta):
    '''Canonizer Base class.
    Canonizers modify modules temporarily such that certain attribution rules can properly be applied.
    '''
    @abstractmethod
    def apply(self, root_module):
        '''Apply this canonizer recursively on all applicable modules.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            Root module to which to apply the canonizers.

        Returns
        -------
        list
            A list of all applied instances of this class.
        '''
        return []

    @abstractmethod
    def register(self):
        '''Apply the changes of this canonizer.'''

    @abstractmethod
    def remove(self):
        '''Revert the changes introduces by this canonizer.'''

    def copy(self):
        '''Return a copy of this instance.'''
        return self.__class__()


class MergeBatchNorm(Canonizer):
    '''Abstract Canonizer to merge the parameters of batch norms into linear modules.'''
    linear_type = (
        Linear,
    )
    batch_norm_type = (
        BatchNorm,
    )

    def __init__(self):
        super().__init__()
        self.linears = None
        self.batch_norm = None

        self.linear_params = None
        self.batch_norm_params = None

    def register(self, linears, batch_norm):
        '''Store the parameters of the linear modules and the batch norm module and apply the merge.

        Parameters
        ----------
        linear: list of obj:`torch.nn.Module`
            List of linear layer with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes
            `running_mean`, `running_var`, `weight`, `bias` and `eps`
        '''
        self.linears = linears
        self.batch_norm = batch_norm

        self.linear_params = [(linear.weight.data, getattr(linear.bias, 'data', None)) for linear in linears]

        self.batch_norm_params = {
            key: getattr(self.batch_norm, key).data for key in ('weight', 'bias', 'running_mean', 'running_var')
        }

        self.merge_batch_norm(self.linears, self.batch_norm)

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        for linear, (weight, bias) in zip(self.linears, self.linear_params):
            linear.weight.data = weight
            if bias is None:
                linear.bias = None
            else:
                linear.bias.data = bias

        for key, value in self.batch_norm_params.items():
            getattr(self.batch_norm, key).data = value

    @staticmethod
    def merge_batch_norm(modules, batch_norm):
        '''Update parameters of a linear layer to additionally include a Batch Normalization operation and update the
        batch normalization layer to instead compute the identity.

        Parameters
        ----------
        modules: list of obj:`torch.nn.Module`
            Linear layers with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes `running_mean`, `running_var`, `weight`, `bias` and
            `eps`
        '''
        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)

        for module in modules:
            original_weight = module.weight.data
            if module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(1, device=original_weight.device, dtype=original_weight.dtype)
                )
            original_bias = module.bias.data

            if isinstance(module, ConvolutionTranspose):
                index = (None, slice(None), *((None,) * (original_weight.ndim - 2)))
            else:
                index = (slice(None), *((None,) * (original_weight.ndim - 1)))

            # merge batch_norm into linear layer
            module.weight.data = (original_weight * scale[index])
            module.bias.data = (original_bias - batch_norm.running_mean) * scale + batch_norm.bias

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)


class SequentialMergeBatchNorm(MergeBatchNorm):
    '''Canonizer to merge the parameters of all batch norms that appear sequentially right after a linear module.

    Note
    ----
    SequentialMergeBatchNorm traverses the tree of children of the provided module depth-first and in-order.
    This means that child-modules must be assigned to their parent module in the order they are visited in the forward
    pass to correctly identify adjacent modules.
    This also means that activation functions must be assigned in their module-form as a child to their parent-module
    to properly detect when there is an activation function between linear and batch-norm modules.

    '''
    def apply(self, root_module):
        '''Finds a batch norm following right after a linear layer, and creates a copy of this instance to merge
        them by fusing the batch norm parameters into the linear layer and reducing the batch norm to the identity.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            A module of which the leaves will be searched and if a batch norm is found right after a linear layer, will
            be merged.

        Returns
        -------
        instances: list
            A list of instances of this class which modified the appropriate leaves.
        '''
        instances = []
        last_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(last_leaf, self.linear_type) and isinstance(leaf, self.batch_norm_type):
                instance = self.copy()
                instance.register((last_leaf,), leaf)
                instances.append(instance)
            last_leaf = leaf

        return instances


class NamedMergeBatchNorm(MergeBatchNorm):
    '''Canonizer to merge the parameters of all batch norms into linear modules, specified by their respective names.

    Parameters
    ----------
    name_map: list[tuple[string], string]
        List of which linear layer names belong to which batch norm name.
    '''
    def __init__(self, name_map):
        super().__init__()
        self.name_map = name_map

    def apply(self, root_module):
        '''Create appropriate merges given by the name map.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            Root module for which underlying modules will be merged.

        Returns
        -------
        instances: list
            A list of merge instances.
        '''
        instances = []
        lookup = dict(root_module.named_modules())

        for linear_names, batch_norm_name in self.name_map:
            instance = self.copy()
            instance.register([lookup[name] for name in linear_names], lookup[batch_norm_name])
            instances.append(instance)

        return instances

    def copy(self):
        return self.__class__(self.name_map)


class AttributeCanonizer(Canonizer):
    '''Canonizer to set an attribute of module instances.
    Note that the use of this Canonizer removes previously set attributes after removal.

    Parameters
    ----------
    attribute_map: Function
        A function that returns either None, if not applicable, or a dict with keys describing which attributes to
        overload for a module. The function signature is (name: string, module: type) -> None or
        dict.
    '''
    def __init__(self, attribute_map):
        self.attribute_map = attribute_map
        self.attribute_keys = None
        self.module = None

    def apply(self, root_module):
        '''Overload the attributes for all applicable modules.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            Root module for which underlying modules will have their attributes overloaded.

        Returns
        -------
        instances : list of obj:`Canonizer`
            The applied canonizer instances, which may be removed by calling `.remove`.
        '''
        instances = []
        for name, module in root_module.named_modules():
            attributes = self.attribute_map(name, module)
            if attributes is not None:
                instance = self.copy()
                instance.register(module, attributes)
                instances.append(instance)
        return instances

    def register(self, module, attributes):
        '''Overload the module's attributes.

        Parameters
        ---------
        module : obj:`torch.nn.Module`
            The module of which the attributes will be overloaded.
        attributes : dict
            The attributes which to overload for the module.
        '''
        self.attribute_keys = list(attributes)
        self.module = module
        for key, value in attributes.items():
            setattr(module, key, value)

    def remove(self):
        '''Remove the overloaded attributes. Note that functions are descriptors, and therefore not direct attributes
        of instance, which is why deleting instance attributes with the same name reverts them to the original
        function.
        '''
        for key in self.attribute_keys:
            delattr(self.module, key)

    def copy(self):
        '''Copy this Canonizer.

        Returns
        -------
        obj:`Canonizer`
            A copy of this Canonizer.
        '''
        return AttributeCanonizer(self.attribute_map)


class CompositeCanonizer(Canonizer):
    '''A Composite of Canonizers, which applies all supplied canonizers.

    Parameters
    ----------
    canonizers : list of obj:`Canonizer`
        Canonizers of which to build a Composite of.
    '''
    def __init__(self, canonizers):
        self.canonizers = canonizers

    def apply(self, root_module):
        '''Apply call canonizers.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            Root module for which underlying modules will have canonizers applied.

        Returns
        -------
        instances : list of obj:`Canonizer`
            The applied canonizer instances, which may be removed by calling `.remove`.
        '''
        instances = []
        for canonizer in self.canonizers:
            instances += canonizer.apply(root_module)
        return instances

    def register(self):
        '''Register this Canonizer. Nothing to do for a CompositeCanonizer.'''

    def remove(self):
        '''Remove this Canonizer. Nothing to do for a CompositeCanonizer.'''
