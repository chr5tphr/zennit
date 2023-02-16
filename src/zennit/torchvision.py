# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/torchvision.py
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
'''Specialized Canonizers for models from torchvision.'''
import torch
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, BasicBlock as ResNetBasicBlock

from .canonizers import SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from .layer import Sum


class VGGCanonizer(SequentialMergeBatchNorm):
    '''Canonizer for torchvision.models.vgg* type models. This is so far identical to a SequentialMergeBatchNorm'''


class ResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out


class ResNetBasicBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out


class ResNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ResNetBottleneckCanonizer(),
            ResNetBasicBlockCanonizer(),
        ))


class DenseNetAdaptiveAvgPoolCanonizer(AttributeCanonizer):
    '''Canonizer specifically for AdaptiveAvgPooling2d layers at the end of torchvision.model densenet models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, DenseNet):
            attributes = {
                'forward': cls.forward.__get__(module),
            }
            return attributes
        return None

    def copy(self):
        '''Copy this Canonizer.

        Returns
        -------
        obj:`Canonizer`
            A copy of this Canonizer.
        '''
        return DenseNetAdaptiveAvgPoolCanonizer()

    def register(self, module, attributes):
        module.features.add_module('final_relu', ReLU(inplace=True))
        module.features.add_module('adaptive_avg_pool2d', AdaptiveAvgPool2d((1, 1)))
        super(DenseNetAdaptiveAvgPoolCanonizer, self).register(module, attributes)

    def remove(self):
        '''Remove the overloaded attributes. Note that functions are descriptors, and therefore not direct attributes
        of instance, which is why deleting instance attributes with the same name reverts them to the original
        function.
        '''
        self.module.features = Sequential(*list(self.module.features.children())[:-2])
        for key in self.attribute_keys:
            delattr(self.module, key)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class DenseNetSeqThreshCanonizer(CompositeCanonizer):
    def __init__(self):
        super().__init__((
            DenseNetAdaptiveAvgPoolCanonizer(),
            SequentialMergeBatchNorm(),
            ThreshReLUMergeBatchNorm(),
        ))


class DenseNetThreshSeqCanonizer(CompositeCanonizer):
    def __init__(self):
        super().__init__((
            DenseNetAdaptiveAvgPoolCanonizer(),
            ThreshReLUMergeBatchNorm(),
            SequentialMergeBatchNorm(),
        ))