# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/cmap.py
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
'''Create color maps from a color-map specification language'''
import re
from typing import NamedTuple

import numpy as np


class CMapToken(NamedTuple):
    '''Tokens used by the lexer of ColorMap.'''
    type: str
    value: str
    pos: int


class ColorNode(NamedTuple):
    '''Nodes produced by the parser of ColorMap.'''
    index: int
    value: np.ndarray


class ColorMap:
    '''Compile a color map from color-map specification language (cmsl) source code.

    The color-map specification language (cmsl) is used to specify linear color maps with comma-separated colors
    supplied as hexadecimal values for each color channel in RGB, with either 1 or 2 values per channel.  Optionally, a
    hexadecimal index with either one or two digits may be supplied in front of each color, followed by a colon, to
    indicate the index which should be the color. Values for the ColorMap in-between colors will be interpolated
    linearly. If no index is supplied, colors without indices will be spaced evenly between indices. If the first and
    last indices are supplied but not 0 (or 00) and f (or ff) respectively, they will be added as an additional node in
    the color map, with the same color as the colors with the lowest and highest index respectively.  If indices are
    provided, they must be in ascending order from left to right, with an arbitrary number of non-indexed colors. If
    the first and/or last color are not indexed, they are assumed to be 0 (or 00) and f (or ff) respectively.

    Parameters
    ----------
    source : str
        Source code to generate the color map.

    '''
    _rexp = re.compile(
        r'(?P<longcolor>[0-9a-fA-F]{6})|'
        r'(?P<shortcolor>[0-9a-fA-F]{3})|'
        r'(?P<index>[0-9a-fA-F]{1,2})|'
        r'(?P<adsep>:)|'
        r'(?P<sep>,)|'
        r'(?P<whitespace>\s+)|'
        r'(?P<error>.+)'
    )

    def __init__(self, source):
        self._source = None
        self.source = source

    @property
    def source(self) -> str:
        '''Source code property used to generate the color map. May be overwritten with a new string, which will be
        compiled to change the color map.
        '''
        return self._source

    @source.setter
    def source(self, value: str):
        '''Set source code property and re-compile the color map.'''
        try:
            tokens = self._lex(value)
            nodes = self._parse(tokens)
            self._indices, self._colors = self._make_palette(nodes)
        except RuntimeError as err:
            raise RuntimeError('Compilation of ColorMap failed!') from err

        self._source = value

    @staticmethod
    def _lex(string):
        '''Lexical scanning of cmsl using regular expressions.'''
        return [CMapToken(match.lastgroup, match.group(), match.start()) for match in ColorMap._rexp.finditer(string)]

    @staticmethod
    def _parse(tokens):
        '''Parse cmsl tokens into a list of color nodes.'''
        nodes = []
        log = []
        for token in tokens:
            if token.type == 'index' and not log:
                log.append(token)
            elif token.type == 'adsep' and len(log) == 1 and log[-1].type == 'index':
                log.append(token)
            elif token.type in ('shortcolor', 'longcolor'):
                if len(log) == 2 and log[-2].type == 'index':
                    indval = log[-2].value
                    if len(indval) == 1:
                        indval = indval * 2
                    index = int(indval, base=16)
                elif not log:
                    index = None
                else:
                    raise RuntimeError(f'Unexpected {token}')

                value_it = iter(token.value) if token.type == 'longcolor' else token.value
                value = [int(''.join(chars), base=16) for chars in zip(*[value_it] * 2)]
                nodes.append(ColorNode(index, np.array(value)))
                log.append(token)
            elif token.type == 'sep' and log and log[-1].type in ('longcolor', 'shortcolor'):
                log.clear()
            elif token.type != 'whitespace':
                raise RuntimeError(f'Unexpected {token}')

        if log and log[-1].type not in ('shortcolor', 'longcolor'):
            raise RuntimeError(f'Unexpected {log[-1]}')

        return nodes

    @staticmethod
    def _make_palette(nodes):
        '''Generate color map indices and colors from a list of color nodes.'''
        if len(nodes) < 2:
            raise RuntimeError("ColorMap needs at least 2 colors!")
        result = []
        log = []

        start = nodes.pop(0)
        result.append(ColorNode(0, start.value))
        if start.index is not None and start.index > 0:
            result.append(start)

        for n, node in enumerate(nodes):
            if node.index is None:
                if n < len(nodes) - 1:
                    log.append(node)
                    continue
                node = ColorNode(255, node.value)
            elif node.index < result[-1].index:
                raise RuntimeError('ColorMap indices not ordered! Provided indices are required in ascending order.')
            if log:
                result += [
                    ColorNode(
                        int(result[-1].index * (1. - alpha) + node.index * alpha),
                        lognode.value
                    ) for alpha, lognode in zip(np.linspace(0., 1., len(log) + 2)[1:-1], log)
                ]
                log.clear()
            result.append(node)

        result.append(ColorNode(256, result[-1].value))

        indices = np.array([node.index for node in result])
        colors = np.stack([node.value for node in result], axis=0)

        return indices, colors

    def __call__(self, x):
        '''Map scalar values in the range [0, 1] to RGB. This appends an axis with size 3 to `x`. Values are clipped to
        the range [0, 1], and the output will also lie in this range.

        Parameters
        ----------
        x : obj:`numpy.ndarray`
            Input array of arbitrary shape, which will be clipped to range [0, 1], and mapped to RGB using this
            ColorMap.

        Returns
        -------
        obj:`numpy.ndarray`
            The input array `x`, clipped to [0, 1] and mapped to RGB given this colormap, where the 3 color channels
            are appended as a new axis to the end.
        '''
        x = (x * 255).clip(0, 255)
        index = np.searchsorted(self._indices[:-1], x, side='right')
        alpha = ((x - self._indices[index - 1]) / (self._indices[index] - self._indices[index - 1]))[..., None]
        return (self._colors[index - 1] * (1 - alpha) + self._colors[index] * alpha) / 255.

    def palette(self, level=1.0):
        '''Create an 8-bit palette.

        Parameters
        ----------
        level: float
            The level of the color map. 1.0 is default. Values below zero reduce the color range, with only a single
            color used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly
            steep transition at the center of the image.

        Returns
        -------
        obj:`numpy.ndarray`
            The palette described by an unsigned 8-bit numpy array with 256 entries.
        '''
        x = np.linspace(-1., 1., 256, dtype=np.float64) * level
        x = ((x + 1.) / 2.).clip(0., 1.)
        x = self(x)
        x = (x * 255.).round(12).clip(0., 255.).astype(np.uint8)
        return x


class LazyColorMapCache:
    '''Dict-like object to store sources for colormaps, and compile and cache them lazily.

    Parameters
    ----------
    sources : dict
        Dict containing a mapping from names to color map specification language source.
    '''
    def __init__(self, sources):
        self._sources = sources
        self._compiled = {}

    def __getitem__(self, name):
        if name not in self._sources:
            raise KeyError(f'No source for key {name}.')
        if name not in self._compiled:
            self._compiled[name] = ColorMap(self._sources[name])
        return self._compiled[name]

    def __setitem__(self, name, value):
        self._sources[name] = value
        if name in self._compiled:
            self._compiled[name].source = value

    def __delitem__(self, name):
        del self._sources[name]
        if name in self._compiled:
            del self._compiled[name]

    def __iter__(self):
        return iter(self._sources)

    def __len__(self):
        return len(self._sources)
