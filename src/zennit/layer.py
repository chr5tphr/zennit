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
        '''Computes the sum along a dimension.

        Parameters
        ----------
        input: :py:obj:`torch.Tensor`
            The input on which to sum.

        Returns
        -------
        :py:obj:`torch.Tensor`
            The resulting tensor summed along dimension `dim`.

        '''
        return torch.sum(input, dim=self.dim)


class PairwiseCentroidDistance(torch.nn.Module):
    '''Compute pairwise distances between inputs and centroids.

    Initialized with a set of centroids, this layer computes the pairwise distance between the input and the centroids.

    Parameters
    ----------
    centroids : :py:obj:`torch.Tensor`
        shape (K, D) tensor of centroids
    power : float
        power to raise the distance to

    Examples
    --------
    >>> centroids = torch.randn(10, 2)
    >>> distance = PairwiseCentroidDistance(centroids)
    >>> x = torch.randn(100, 2)
    >>> distance(x)

    '''
    def __init__(self, centroids, power=2):
        super().__init__()
        self.centroids = torch.nn.Parameter(centroids)
        self.power = power

    def forward(self, input):
        '''Computes the pairwise distance between `input` and `self.centroids` and raises to the power `self.power`.

        Parameters
        ----------
        input : :py:obj:`torch.Tensor`
            shape (N, D) tensor of points

        Returns
        -------
        :py:obj:`torch.Tensor`
            shape (N, K) tensor of distances
        '''
        return torch.cdist(input, self.centroids)**self.power


class NeuralizedKMeans(torch.nn.Module):
    '''Compute the k-means discriminants for a set of points.

    Technically, this is a tensor-matrix product with a bias.

    Parameters
    ----------
    weight : :py:obj:`torch.Tensor`
        shape (K, K-1, D) tensor of weights
    bias : :py:obj:`torch.Tensor`
        shape (K, K-1) tensor of biases

    Examples
    --------
    >>> weight = torch.randn(10, 9, 2)
    >>> bias = torch.randn(10, 9)
    >>> neuralized_kmeans = NeuralizedKMeans(weight, bias)

    '''
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        '''Computes the tensor-matrix product of `x` and `self.weight` and adds `self.bias`.

        Parameters
        ----------
        x : :py:obj:`torch.Tensor`
            shape (N, D) tensor of points

        Returns
        -------
        :py:obj:`torch.Tensor`
            shape (N, K, K-1) tensor of k-means discriminants
        '''
        return torch.einsum('nd,kjd->nkj', x, self.weight) + self.bias


class MinPool2d(torch.nn.MaxPool2d):
    '''Computes a min pool.

    Parameters
    ----------
    kernel_size : int or tuple
        size of the pooling window
    stride : int or tuple
        stride of the pooling operation
    padding : int or tuple
        zero-padding added to both sides of the input
    dilation : int or tuple
        spacing between kernel elements
    return_indices : bool
        if True, will return the max indices along with the outputs
    ceil_mode : bool
        if True, will use ceil instead of floor to compute the output shape

    Examples
    --------
    >>> pool = MinPool2d(2)
    >>> x = torch.randn(1, 1, 4, 4)
    >>> pool(x)
    '''
    def forward(self, input):
        '''Computes the min pool of `input`.

        Parameters
        ----------
        input : :py:obj:`torch.Tensor`
            the input tensor

        Returns
        -------
        :py:obj:`torch.Tensor`
            the min pool of `input`
        n_dims = input.shape[self.dim]'''
        return -super().forward(-input)


class MinPool1d(torch.nn.MaxPool1d):
    '''Computes a min pool.

    Parameters
    ----------
    kernel_size : int or tuple
        size of the pooling window
    stride : int or tuple
        stride of the pooling operation
    padding : int or tuple
        zero-padding added to both sides of the input
    dilation : int or tuple
        spacing between kernel elements
    return_indices : bool
        if True, will return the max indices along with the outputs
    ceil_mode : bool
        if True, will use ceil instead of floor to compute the output shape

    Examples
    --------
    >>> pool = MinPool1d(2)
    >>> x = torch.randn(1, 1, 4)
    >>> pool(x)
    '''
    def forward(self, input):
        '''Computes the min pool of `input`.

        Parameters
        ----------
        input : :py:obj:`torch.Tensor`
            the input tensor

        Returns
        -------
        :py:obj:`torch.Tensor`
            the min pool of `input`
        '''
        return -super().forward(-input)
