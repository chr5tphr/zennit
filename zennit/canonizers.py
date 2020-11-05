'''Functions to produce a canonical form of models fit for LRP'''
from abc import ABCMeta, abstractmethod

import torch

from .core import collect_leaves
from .types import Linear, BatchNorm


class Canonizer(metaclass=ABCMeta):
    '''Canonizer Base class.
    Canonizers modify modules temporarily such that certain attribution rules can properly be applied.
    '''
    @abstractmethod
    def apply(self, module):
        '''Apply this canonizer recursively on all applicable modules.

        Parameters
        ----------
        module: obj:`torch.nn.Module`
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
        return self.__class__()


class MergeBatchNorm(Canonizer):
    '''Canonizer to merge the parameters of all batch norms that appear sequentially right after a linear module.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        Linear layer with mandatory attributes `weight` and `bias`.
    batch_norm: obj:`torch.nn.Module`
        Batch Normalization module with mandatory attributes `running_mean`, `running_var`, `weight`, `bias` and `eps`
    '''
    linear_type = (
        Linear,
    )
    batch_norm_type = (
        BatchNorm,
    )

    def __init__(self):
        self.linears = None
        self.batch_norm = None

        self.linear_params = None
        self.batch_norm_params = None

    def register(self, linears, batch_norm):
        '''Store the parameters of the linear modules and the batch norm module and apply the merge.'''
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

    def apply(self, module):
        '''Finds a batch norm following right after a linear layer, and creates a copy of this instance to merge
        them by fusing the batch norm parameters into the linear layer and reducing the batch norm to the identity.

        Parameters
        ----------
        module: obj:`torch.nn.Module`
            A module of which the leaves will be searched and if a batch norm is found right after a linear layer, will
            be merged.

        Returns
        -------
        instances: list
            A list of instances of this class which modified the appropriate leaves.
        '''
        instances = []
        last_leaf = None
        for leaf in collect_leaves(module):
            if isinstance(last_leaf, self.linear_type) and isinstance(leaf, self.batch_norm_type):
                instance = self.copy()
                instance.register((last_leaf,), leaf)
                instances.append(instance)
            last_leaf = leaf

        return instances

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

            # merge batch_norm into linear layer
            module.weight.data = (original_weight * scale[:, None, None, None])
            module.bias.data = (original_bias - batch_norm.running_mean) * scale + batch_norm.bias

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)
