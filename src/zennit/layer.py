# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/layer.py
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
'''Additional Utility Layers'''
import torch


class Sum(torch.nn.Module):
    '''Compute the sum along an axis.

    Parameters
    ----------
    dim : int
        Dimension over which to sum.
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        '''Computes the sum along a dimension.'''
        return torch.sum(input, dim=self.dim)
