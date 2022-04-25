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
    # blue to white to red as in the french flag
    'france': '0055a4,80:fff,ef4135',
    # blue to white to red with brightness 0xd0
    'seismic': '0000d0,80:d0d0d0,d00000',
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
    if cmap in CMAPS:
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


def imgify(obj, vmin=None, vmax=None, cmap='bwr', level=1.0, symmetric=False, grid=False, gridfill=None):
    '''Convert an array with 1 or 3 channels to a PIL image.
    The color dimension can be either the first or the last dimension.

    Parameters
    ----------
    obj: object
        Anything that can be converted to a numpy array with 2 dimensions greyscale, or 3 dimensions with 1 or 3 values
        in the first or last dimension (color).
    vmin: float or obj:`numpy.ndarray`
        Manual minimum value of the array. Overrides the used norm's minimum value.
    vmax: float or obj:`numpy.ndarray`
        Manual maximum value of the array. Overrides the used norm's maximum value.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette. The color map will only be applied for arrays with only a single color channel.
        The color will be specified as a palette in the PIL Image.
    level: float
        The level of the color map. 1.0 is default. Values below 1.0 reduce the color range, with only a single color
        used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly steep
        transition at the center of the pixel value distribution.
    symmetric : bool, optional
        Specifies whether the norm should be symmetric (True) or unaligned (False, default). If True, normalize with
        both minimum and maximum by the absolute maximum, which will cause 0. in the input to correspond to 0.5 in the
        result. Setting ``symmetric=False`` (default) will result in the minimum value to be directly mapped to 0 and
        the maximum value to be directly mapped to 1. ``vmin`` and ``vmax`` may be used to manually override the
        minimum and maximum value respectively.
    grid : bool or tuple of ints of size 2
        If true, assumes the first dimension to be the batch dimension. If True, creates a square grid of images in the
        batch dimension after normalizing each sample. If tuple of ints of size 2, creates the grid in the shape of
        ``(height, width)``. If False (default), does not assume a batch dimension.
    gridfill: :py:obj:`np.uint8`
        A value to fill empty grid members. Default is the mean pixel value. No effect when ``grid=False``.

    Returns
    -------
    image: obj:`PIL.Image`
        The array visualized as a Pillow Image.
    '''
    array = np.array(obj)

    if grid:
        if isinstance(grid, (list, tuple)) and len(grid) != 2:
            raise TypeError('Grid shape needs to be of size 2!')

        if array.ndim not in (3, 4):
            raise TypeError('Grid input has to have either 3 or 4 axes!')

        if (array.ndim == 4) and (array.shape[3] not in (1, 3)):
            if array.shape[1] in (1, 3):
                array = array.transpose(0, 2, 3, 1)
            else:
                raise TypeError(
                    'After batch, last (or first) axis of input are color channels, '
                    'which have to either be 1, 3 or be omitted entirely!'
                )
    else:
        if array.ndim not in (2, 3):
            raise TypeError('Input has to have either 2 or 3 axes!')

        if (array.ndim == 3) and (array.shape[2] not in (1, 3)):
            if array.shape[0] in (1, 3):
                array = array.transpose(1, 2, 0)
            else:
                raise TypeError(
                    'Last (or first) axis of input are color channels, '
                    'which have to either be 1, 3 or be omitted entirely!'
                )

    # renormalize data if necessary
    if array.dtype != np.uint8:
        if grid:
            dims = tuple(range(1, array.ndim))
        else:
            dims = tuple(range(array.ndim))

        if None in (vmin, vmax):
            vmin_, vmax_ = interval_norm_bounds(array, symmetric=symmetric, dim=dims)

        if vmin is None:
            vmin = vmin_
        else:
            vmin = np.array(vmin)

        if vmax is None:
            vmax = vmax_
        else:
            vmax = np.array(vmax)

        array = (array - vmin) / (vmax - vmin)
        array = (array * 255).clip(0, 255).astype(np.uint8)

    if grid:
        shape = None if isinstance(grid, bool) else grid
        # gridify adds the missing axis
        array = gridify(array, shape=shape, fill_value=gridfill)
    else:
        # add missing axis if omitted
        if array.ndim == 2:
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
        An object that can be converted to a numpy array, with 3 (greyscale) or 4 (rgb) axes.
        The color channel's position is automatically detected, and moved to the back of the shape.
    shape: tuple of size 2, optional
        Height and width of the produced grid. If None (default), create a square grid.
    fill_value: float or obj:`numpy.ndarray`
        A value to fill empty grid members. Default is the mean pixel value.

    Returns
    -------
    obj:`numpy.ndarray`
        An array with the 0-th dimension absorbed into the height and width dimensions (then 0 and 1).
        The color dimension will be the last dimension, even if it was the first dimension before.
    '''
    array = np.array(obj)
    if array.ndim not in (3, 4):
        raise TypeError('For creating an image grid, the array has to have either 3 (greyscale) or 4 (rgb) axes!')

    # add missing axis if omitted
    if array.ndim == 3:
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
    else:
        fill_value = np.array(fill_value).astype(array.dtype)

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


def imsave(
    fp,
    obj,
    vmin=None,
    vmax=None,
    cmap='bwr',
    level=1.0,
    grid=False,
    format=None,
    writer_params=None,
    symmetric=False,
    gridfill=None,
):
    '''Convert an array to an image and save it using file `fp`.
    Internally, `imgify` is called to create a PIL Image, which is then saved using PIL.

    Parameters
    ----------
    fp: str, obj:`pathlib.Path` or file
        Save target for PIL Image.
    obj: object
        Anything that can be converted to a numpy array with 2 dimensions greyscale, or 3 dimensions with 1 or 3 values
        in the first or last dimension (color).
    vmin: float or obj:`numpy.ndarray`
        Manual minimum value of the array. Overrides the used norm's minimum value.
    vmax: float or obj:`numpy.ndarray`
        Manual maximum value of the array. Overrides the used norm's maximum value.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette. The color map will only be applied for arrays with only a single color channel.
        The color will be specified as a palette in the PIL Image.
    level: float
        The level of the color map. 1.0 is default. Values below 1.0 reduce the color range, with only a single color
        used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly steep
        transition at the center of the pixel value distribution.
    grid : bool or tuple of ints of size 2
        If true, assumes the first dimension to be the batch dimension. If True, creates a square grid of images in the
        batch dimension after normalizing each sample. If tuple of ints of size 2, creates the grid in the shape of
        ``(height, width)``. If False (default), does not assume a batch dimension.
    format: str
        Optional format override for PIL Image.save.
    writer_params: dict
        Extra params to the image writer in PIL.
    symmetric : bool, optional
        Specifies whether the norm should be symmetric (True) or unaligned (False, default). If True, normalize with
        both minimum and maximum by the absolute maximum, which will cause 0. in the input to correspond to 0.5 in the
        result. Setting ``symmetric=False`` (default) will result in the minimum value to be directly mapped to 0 and
        the maximum value to be directly mapped to 1. ``vmin`` and ``vmax`` may be used to manually override the
        minimum and maximum value respectively.
    gridfill: :py:obj:`np.uint8`
        A value to fill empty grid members. Default is the mean pixel value. No effect when ``grid=False``.
    '''
    if writer_params is None:
        writer_params = {}
    image = imgify(obj, vmin=vmin, vmax=vmax, cmap=cmap, level=level, symmetric=symmetric, grid=grid, gridfill=gridfill)
    image.save(fp, format=format, **writer_params)


def interval_norm_bounds(input, symmetric=False, dim=None):
    '''Return the boundaries to normalize the data interval batch-wise between 0. and 1. given the specified strategy.

    Parameters
    ----------
    input : :py:class:`numpy.ndarray`
        Array for which to return the boundaries.
    symmetric : bool, optional
        Specifies whether the norm should be symmetric (True) or unaligned (False, default).
        If True, normalize with both minimum and maximum by the absolute maximum, which will cause 0. in the
        input to correspond to 0.5 in the result. Setting ``symmetric=False`` (default) will result in the minimum
        value to be directly mapped to 0 and the maximum value to be directly mapped to 1.
    dim : tuple of ints, optional
        Set the channel dimensions over which the boundaries are computed (default is ``tuple(range(1, input.ndim))``.

    Returns
    -------
    :py:class:`numpy.ndarray`
        The normalized array ``input`` along ``dim`` to lie in the interval [0, 1].

    '''
    if dim is None:
        dim = tuple(range(1, input.ndim))

    if symmetric:
        # 0-aligned symmetric input, negative and positive can be compared, the original 0. becomes 0.5
        vmax = np.abs(input).max(dim, keepdims=True)
        vmin = -vmax
    else:
        # do not align, the original minimum value becomes 0., the original maximum becomes 1.
        vmax = input.max(dim, keepdims=True)
        vmin = input.min(dim, keepdims=True)
    return vmin, vmax
