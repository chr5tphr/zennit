'''Functions to produce a canonical form of models fit for LRP'''
from abc import ABCMeta, abstractmethod

import torch

from .core import collect_leaves


class Canonizer(metaclass=ABCMeta):
    '''Canonizer Base class.
    Canonizers modify modules temporarily such that certain attribution rules can properly be applied.
    '''
    @classmethod
    @abstractmethod
    def apply(cls, module):
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
        torch.nn.modules.conv.Conv1d,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.conv.Conv3d,
        torch.nn.modules.conv.ConvTranspose1d,
        torch.nn.modules.conv.ConvTranspose2d,
        torch.nn.modules.conv.ConvTranspose3d,
        torch.nn.modules.linear.Linear,
    )
    batch_norm_type = (
        torch.nn.modules.batchnorm.BatchNorm1d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.batchnorm.BatchNorm3d,
    )

    def __init__(self, linear, batch_norm):
        self.linear = linear
        self.batch_norm = batch_norm

        self.linear_weight = None
        self.linear_bias = None

        self.weight = None
        self.bias = None
        self.running_mean = None
        self.running_var = None

        self.register()

    def register(self):
        '''Store the parameters of the linear and the batch norm modules and apply the merge.'''
        self.linear_weight = self.linear.weight.data
        if self.linear.bias is not None:
            self.linear_bias = self.linear.bias.data

        self.weight = self.batch_norm.weight.data
        self.bias = self.batch_norm.bias.data
        self.running_mean = self.batch_norm.running_mean.data
        self.running_var = self.batch_norm.running_var.data

        self.merge_batch_norm(self.linear, self.batch_norm)

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to what they were
        before the merge.'''
        self.linear.weight.data = self.linear_weight

        if self.linear_bias is None:
            self.linear.bias = None
        else:
            self.linear.bias.data = self.linear_bias

        self.batch_norm.weight.data = self.weight
        self.batch_norm.bias.data = self.bias
        self.batch_norm.running_mean.data = self.running_mean
        self.batch_norm.running_var.data = self.running_var

    @classmethod
    def apply(cls, module):
        '''Finds a batch norm following right after a linear layer, and creates an instance of this class to merge
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
            if isinstance(last_leaf, cls.linear_type) and isinstance(leaf, cls.batch_norm_type):
                instances.append(cls(last_leaf, leaf))
            last_leaf = leaf

        return instances

    @staticmethod
    def merge_batch_norm(module, batch_norm):
        '''Update parameters of a linear layer to additionally include a Batch Normalization operation and update the
        batch normalization layer to instead compute the identity.

        Parameters
        ----------
        module: obj:`torch.nn.Module`
            Linear layer with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes `running_mean`, `running_var`, `weight`, `bias` and
            `eps`
        '''
        original_weight = module.weight.data
        if module.bias is None:
            module.bias = torch.nn.Parameter(torch.zeros(1, device=original_weight.device, dtype=original_weight.dtype))
        original_bias = module.bias.data

        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)

        # merge batch_norm into linear layer
        module.weight.data = (original_weight * scale[:, None, None, None])
        module.bias.data = (original_bias - batch_norm.running_mean) * scale + batch_norm.bias

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)
