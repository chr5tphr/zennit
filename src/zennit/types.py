# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/types.py
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
'''Type definitions for convenience.'''
import torch


class SubclassMeta(type):
    '''Meta class to bundle multiple subclasses.'''
    def __instancecheck__(cls, inst):
        """Implement isinstance(inst, cls) as subclasscheck."""
        return cls.__subclasscheck__(type(inst))

    def __subclasscheck__(cls, sub):
        """Implement issubclass(sub, cls) with by considering additional __subclass__ members."""
        candidates = cls.__dict__.get("__subclass__", tuple())
        return type.__subclasscheck__(cls, sub) or issubclass(sub, candidates)


class ConvolutionTranspose(metaclass=SubclassMeta):
    '''Abstract base class that describes transposed convolutional modules.'''
    __subclass__ = (
        torch.nn.modules.conv.ConvTranspose1d,
        torch.nn.modules.conv.ConvTranspose2d,
        torch.nn.modules.conv.ConvTranspose3d,
    )


class ConvolutionStandard(metaclass=SubclassMeta):
    '''Abstract base class that describes standard (forward) convolutional modules.'''
    __subclass__ = (
        torch.nn.modules.conv.Conv1d,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.conv.Conv3d,
    )


class Convolution(metaclass=SubclassMeta):
    '''Abstract base class that describes all convolutional modules.'''
    __subclass__ = (
        ConvolutionStandard,
        ConvolutionTranspose,
    )


class Linear(metaclass=SubclassMeta):
    '''Abstract base class that describes linear modules.'''
    __subclass__ = (
        Convolution,
        torch.nn.modules.linear.Linear,
    )


class BatchNorm(metaclass=SubclassMeta):
    '''Abstract base class that describes batch normalization modules.'''
    __subclass__ = (
        torch.nn.modules.batchnorm.BatchNorm1d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.batchnorm.BatchNorm3d,
    )


class AvgPool(metaclass=SubclassMeta):
    '''Abstract base class that describes sum-pooling modules.'''
    __subclass__ = (
        torch.nn.modules.pooling.AvgPool1d,
        torch.nn.modules.pooling.AvgPool2d,
        torch.nn.modules.pooling.AvgPool3d,
        torch.nn.modules.pooling.AdaptiveAvgPool1d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool3d,
    )


class MaxPool(metaclass=SubclassMeta):
    '''Abstract base class that describes max-pooling modules.'''
    __subclass__ = (
        torch.nn.modules.pooling.MaxPool1d,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.pooling.MaxPool3d,
        torch.nn.modules.pooling.AdaptiveMaxPool1d,
        torch.nn.modules.pooling.AdaptiveMaxPool2d,
        torch.nn.modules.pooling.AdaptiveMaxPool3d,
    )


class Activation(metaclass=SubclassMeta):
    '''Abstract base class that describes activation modules.'''
    __subclass__ = (
        torch.nn.modules.activation.ELU,
        torch.nn.modules.activation.Hardshrink,
        torch.nn.modules.activation.Hardsigmoid,
        torch.nn.modules.activation.Hardtanh,
        torch.nn.modules.activation.Hardswish,
        torch.nn.modules.activation.LeakyReLU,
        torch.nn.modules.activation.LogSigmoid,
        torch.nn.modules.activation.MultiheadAttention,
        torch.nn.modules.activation.PReLU,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.activation.ReLU6,
        torch.nn.modules.activation.RReLU,
        torch.nn.modules.activation.SELU,
        torch.nn.modules.activation.CELU,
        torch.nn.modules.activation.GELU,
        torch.nn.modules.activation.Sigmoid,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.activation.Softplus,
        torch.nn.modules.activation.Softshrink,
        torch.nn.modules.activation.Softsign,
        torch.nn.modules.activation.Tanh,
        torch.nn.modules.activation.Tanhshrink,
        torch.nn.modules.activation.Threshold,
    )
