# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/image.py
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
'''Functionality to convert arrays to images'''
import numpy as np
from PIL import Image

from .cmap import LazyColorMapCache, ColorMap


# CMAPS contains all built-in color maps
CMAPS = LazyColorMapCache({
    # black to white
    'gray': '000,fff',
    # white to red
    'wred': 'fff,f00',
    # white to blue
    'wblue': 'fff,00f',
    # black to red to yellow to white
    'hot': '000,f00,ff0,fff',
    # black to blue to cyan
    'cold': '000,00f,0ff',
    # combination of cold (reversed) and hot, centered around black
    'coldnhot': '0ff,00f,80:000,f00,ff0,fff',
    # combination of wblue (reversed) and wred, centered around white
    'bwr': '00f,80:fff,f00',
    # red to white to blue as in the french flag
    'france': '0055a4,80:fff,ef4135',
    # red to white to blue with brightness 0xd0
    'seismic': 'd00000,80:d0d0d0,0000d0',
    # cyan to white to magenta with brightness 0xd0
    'coolio': '00d0d0,80:d0d0d0,d000d0',
    # green to white to magenta with brightness 0xd0
    'coleus': '00d000,80:d0d0d0,d000d0',
})


def get_cmap(cmap):
    '''Convenience function to lookup built-in color maps, or create color maps from a source code.

    Parameters
    ----------
    cmap : str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance.

    Returns
    -------
    ColorMap
        The built-in color map with key `cmap` in CMAPS, a new color map created from the code `cmap`, or `cmap` if it
        already was a ColorMap.
    '''
    if isinstance(cmap, ColorMap):
        return cmap
    elif cmap in CMAPS:
        return CMAPS[cmap]
    return ColorMap(cmap)


def palette(cmap='bwr', level=1.0):
    '''Convenience function to create palettes from built-in colormaps, or from a source code if necessary.

    Parameters
    ----------
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette.
    level: float
        The level of the color map palette. 1.0 is default. Values below zero reduce the color range, with only a
        single color used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an
        increasingly steep transition at the center of the color map range.

    Returns
    -------
    obj:`numpy.ndarray`
        The palette described by an unsigned 8-bit numpy array with 256 entries.
    '''
    colormap = get_cmap(cmap)
    return colormap.palette(level=level)


def imgify(obj, vmin=None, vmax=None, cmap='bwr', level=1.0):
    '''Convert an array with 1 or 3 channels to a PIL image.
    The color dimension can be either the first or the last dimension.

    Parameters
    ----------
    obj: object
        Anything that can be converted to a numpy array with 2 dimensions grayscale, or 3 dimensions with 1 or 3 values
        in the first or last dimension (color).
    vmin: float or obj:`numpy.ndarray`
        Minimum value of the array.
    vmax: float or obj:`numpy.ndarray`
        Maximum value of the array.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette. The color map will only be applied for arrays with only a single color channel.
        The color will be specified as a palette in the PIL Image.
    level: float
        The level of the color map. 1.0 is default. Values below zero reduce the color range, with only a single color
        used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly steep
        transition at the center of the image.

    Returns
    -------
    image: obj:`PIL.Image`
        The array visualized as a Pillow Image.
    '''
    try:
        array = np.array(obj)
    except TypeError as err:
        raise TypeError('Could not cast instance of \'{}\' to numpy array.'.format(str(type(obj)))) from err

    if len(array.shape) not in (2, 3):
        raise TypeError('Input has to have either 2 or 3 axes!')

    if (len(array.shape) == 3) and (array.shape[2] not in (1, 3)):
        if array.shape[0] in (1, 3):
            array = array.transpose(1, 2, 0)
        else:
            raise TypeError(
                'Last (or first) axis of input are color channels, '
                'which have to either be 1, 3 or be omitted entirely!'
            )

    # renormalize data if necessary
    if array.dtype != np.uint8:
        if vmin is None:
            vmin = array.min()
        if vmax is None:
            vmax = array.max()
        array = ((array - vmin) * 255 / (vmax - vmin)).clip(0, 255).astype(np.uint8)

    # add missing axis if omitted
    if len(array.shape) == 2:
        array = array[:, :, None]

    # apply palette if single channel
    if array.shape[2] == 1:
        image = Image.fromarray(array[..., 0], mode='P')
        image.putpalette(palette(cmap, level))
    else:
        image = Image.fromarray(array, mode='RGB')

    return image


def gridify(obj, shape=None, fill_value=None):
    '''Align multiple arrays, described as an additional 0-th dimension, into a grid with the 0-th dimension removed.

    Parameters
    ----------
    obj: object
        An object that can be converted to a numpy array, with 3 (grayscale) or 4 (rgb) axes.
        The color channel's position is automatically detected, and moved to the back of the shape.
    shape: tuple of size 2, optional
        Height and width of the produced grid. If None (default), create a square grid.
    fill_value: float or obj:`numpy.ndarray`
        A value to fill empty grid members. May be any compatible shape to `obj`.

    Returns
    -------
    obj:`numpy.ndarray`
        An array with the 0-th dimension absorbed into the height and width dimensions (then 0 and 1).
        The color dimension will be the last dimension, even if it was the first dimension before.
    '''
    try:
        array = np.array(obj)
    except TypeError as err:
        raise TypeError('Could not cast instance of \'{}\' to numpy array.'.format(str(type(obj)))) from err
    if len(array.shape) not in (3, 4):
        raise TypeError('For creating an image grid, the array has to have either 3 (grayscale) or 4 (rgb) axes!')

    # add missing axis if omitted
    if len(array.shape) == 3:
        array = array[..., None]

    if array.shape[3] not in (1, 3):
        if array.shape[1] in (1, 3):
            array = array.transpose(0, 2, 3, 1)
        else:
            raise TypeError(
                'Last (or first) axis of input are color channels, '
                'which have to either be 1, 3 or be omitted entirely!'
            )

    num, height, width, channels = array.shape

    if shape is None:
        grid_width = int(num ** 0.5)
        grid_height = (num + grid_width - 1) // grid_width
    else:
        grid_height, grid_width = shape

    if fill_value is None:
        fill_value = array.min((0, 1, 2), keepdims=True)

    dim = min(num, grid_height * grid_width)
    result = np.zeros((grid_height * grid_width, height, width, channels), dtype=array.dtype) + fill_value
    result[:dim] = array[:dim]
    result = (
        result
        .reshape(grid_height, grid_width, height, width, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_height * height, grid_width * width, channels)
    )

    return result


def imsave(fp, obj, vmin=None, vmax=None, cmap='bwr', level=1.0, grid=False, format=None, writer_params=None):
    '''Convert an array to an image and save it using file `fp`.
    Internally, `imgify` is called to create a PIL Image, which is then saved using PIL.

    Parameters
    ----------
    fp: str, obj:`pathlib.Path` or file
        Save target for PIL Image.
    obj: object
        Anything that can be converted to a numpy array with 2 dimensions grayscale, or 3 dimensions with 1 or 3 values
        in the first or last dimension (color).
    vmin: float or obj:`numpy.ndarray`
        Minimum value of the array.
    vmax: float or obj:`numpy.ndarray`
        Maximum value of the array.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette. The color map will only be applied for arrays with only a single color channel.
        The color will be specified as a palette in the PIL Image.
    level: float
        The level of the color map. 1.0 is default. Values below zero reduce the color range, with only a single color
        used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly steep
        transition at the center of the image.
    grid: bool
        If True, align multiple arrays (in dimension 0) into a grid before creating an image.
    format: str
        Optional format override for PIL Image.save.
    writer_params: dict
        Extra params to the image writer in PIL.
    '''
    if writer_params is None:
        writer_params = {}
    if grid:
        obj = gridify(obj)
    image = imgify(obj, vmin=vmin, vmax=vmax, cmap=cmap, level=level)
    image.save(fp, format=format, **writer_params)
