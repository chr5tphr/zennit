'''Tests for image operations.'''
from typing import NamedTuple
from itertools import product
from io import BytesIO

import pytest
import numpy as np
from PIL import Image

from zennit.cmap import ColorMap
from zennit.image import get_cmap, palette, imgify, gridify, imsave, interval_norm_bounds


@pytest.fixture(scope='session', params=[
    'gray', '000,fff', ColorMap('000,fff')
])
def cmap_source(request):
    '''Fixture for multiple ways to specify the "gray" color map.'''
    return request.param


class ImageTuple(NamedTuple):
    '''NamedTuple for image-array setups.'''
    grid: bool
    nchannels: list
    channel_front: bool
    width: int
    height: int
    array: np.ndarray


@pytest.fixture(scope='session', params=product(
    [False, True],
    [1, 3],
    [False, True],
    [5, 10],
    [5, 10],
    [np.float64, np.uint8]
))
def image_tuple(request):
    '''Image-array setups with varying size, type, number of channels, channel position and grid dimension.'''
    grid, nchannels, channel_front, width, height, dtype = request.param

    shape = (height, width)
    if channel_front:
        shape = (nchannels,) + shape
    else:
        shape = shape + (nchannels,)
    shape = (1,) * grid + shape

    return ImageTuple(
        grid,
        nchannels,
        channel_front,
        width,
        height,
        np.ones(shape, dtype=dtype),
    )


def test_get_cmap(cmap_source):
    '''Test whether get_cmap handles its supported cmap types correctly.'''
    cmap = get_cmap(cmap_source)
    assert isinstance(cmap, ColorMap), 'Returned object is not a ColorMap!'
    assert cmap.source == '000,fff', 'Mismatch in source code of returned ColorMap instance.'


def test_palette(cmap_source):
    '''Test whether palette returns the correct palette for all of its supported types.'''
    pal = palette(cmap_source)
    expected_pal = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    assert np.allclose(expected_pal, pal)


@pytest.mark.parametrize('ndim', [1, 4, 5, 6])
def test_imgify_wrong_dim(ndim):
    '''Test whether imgify fails for an unsupported number of dimensions.'''
    with pytest.raises(TypeError):
        imgify(np.zeros((1,) * ndim))


@pytest.mark.parametrize('ndim', [1, 2, 5, 6])
def test_imgify_grid_wrong_dim(ndim):
    '''Test whether imgify fails for an unsupported number of dimensions with grid=True.'''
    with pytest.raises(TypeError):
        imgify(np.zeros((1,) * ndim), grid=True)


@pytest.mark.parametrize('grid', [[1], (1,), 1, [1, 1, 1], (1, 1, 1)])
def test_imgify_grid_bad_grid(grid):
    '''Test whether imgify fails for unsupported grid values.'''
    with pytest.raises(TypeError):
        imgify(np.zeros((1,) * 4), grid=grid)


@pytest.mark.parametrize('grid,nchannels', product([False, True], [2, 4]))
def test_imgify_wrong_channels(grid, nchannels):
    '''Test whether imgify fails for an unsupported number of dimensions with grid=True.'''
    with pytest.raises(TypeError):
        imgify(np.zeros((1,) * grid + (2, 2, nchannels)), grid=grid)


def test_imgify_container(image_tuple):
    '''Test whether imgify produces the correct PIL Image container'''
    image = imgify(image_tuple.array, grid=image_tuple.grid)
    assert image.mode == ('P' if image_tuple.nchannels == 1 else 'RGB'), 'Mode mismatch!'
    assert image.width == image_tuple.width, 'Width mismatch!'
    assert image.height == image_tuple.height, 'Height mismatch!'


@pytest.mark.parametrize('vmin,vmax,symmetric', product([None, 1.], [None, 2.], [False, True]))
def test_imgify_normalization(vmin, vmax, symmetric):
    '''Test whether imgify normalizes as expected.'''
    array = np.array([[-1., 0., 3.]])

    image = imgify(array, cmap='gray', vmin=vmin, vmax=vmax, symmetric=symmetric)

    if vmin is None:
        if symmetric:
            vmin = -np.abs(array).max()
        else:
            vmin = array.min()
    if vmax is None:
        if symmetric:
            vmax = np.abs(array).max()
        else:
            vmax = array.max()

    expected = (((array - vmin) / (vmax - vmin)) * 255.).clip(0, 255).astype(np.uint8)

    assert np.allclose(np.array(image), expected)


@pytest.mark.parametrize('ndim', [1, 2, 5, 6])
def test_gridify_wrong_dim(ndim):
    '''Test whether imgify fails for an unsupported number of dimensions.'''
    with pytest.raises(TypeError):
        gridify(np.zeros((1,) * ndim))


@pytest.mark.parametrize('channel_front,nchannels', product([False, True], [2, 4]))
def test_gridify_wrong_channels(channel_front, nchannels):
    '''Test whether gridify fails for an unsupported number of channels in both channel positions.'''
    shape = (2, 2)
    if channel_front:
        shape = (nchannels,) + shape
    else:
        shape = shape + (nchannels,)
    shape = (1,) + shape

    with pytest.raises(TypeError):
        gridify(np.zeros(shape))


@pytest.mark.parametrize('shape,expected_shape', [
    [(4, 2, 2, 3), (4, 4, 3)],
    [(4, 2, 2, 1), (4, 4, 1)],
    [(4, 2, 2), (4, 4, 1)],
    [(4, 3, 2, 2), (4, 4, 3)],
    [(4, 1, 2, 2), (4, 4, 1)],
])
def test_gridify_shape(shape, expected_shape):
    '''Test whether gridify produces the correct shape.'''
    output = gridify(np.zeros(shape))
    assert expected_shape == output.shape


@pytest.mark.parametrize('fill_value', [None, 0.])
def test_gridify_fill(fill_value):
    '''Test whether gridify fills empty pixels with the correct value.'''
    array = np.array([[[[1.]]]])
    output = gridify(array, fill_value=fill_value, shape=(1, 2))
    expected_value = array.min() if fill_value is None else fill_value
    assert output[0, 1, 0] == expected_value


@pytest.mark.parametrize('writer_params', [None, {}])
def test_imsave_container(image_tuple, writer_params):
    '''Test whether imsave produces a file, which loads as the correct PIL Image container.'''
    fp = BytesIO()
    imsave(fp, image_tuple.array, grid=image_tuple.grid, format='png', writer_params=writer_params)
    fp.seek(0)
    image = Image.open(fp)
    assert image.mode == ('P' if image_tuple.nchannels == 1 else 'RGB'), 'Mode mismatch!'
    assert image.width == image_tuple.width, 'Width mismatch!'
    assert image.height == image_tuple.height, 'Height mismatch!'


@pytest.mark.parametrize('symmetric,dim,expected_bounds', [
    (False, None, (np.array([[[[-1.]]], [[[0.]]]]), np.array([[[[-0.2]]], [[[0.8]]]]))),
    (False, (1, 2, 3), (np.array([[[[-1.]]], [[[0.]]]]), np.array([[[[-0.2]]], [[[0.8]]]]))),
    (False, (0, 1, 2, 3), (np.array([[[[-1.]]]]), np.array([[[[0.8]]]]))),
    (True, None, (np.array([[[[-1.]]], [[[-0.8]]]]), np.array([[[[1.]]], [[[0.8]]]]))),
    (True, (1, 2, 3), (np.array([[[[-1.]]], [[[-0.8]]]]), np.array([[[[1.]]], [[[0.8]]]]))),
    (True, (0, 1, 2, 3), (np.array([[[[-1.]]]]), np.array([[[[1.]]]]))),
])
def test_interval_norm_bounds(symmetric, dim, expected_bounds):
    '''Test whether interval_norm_bounds computes the correct minimum and maximum values.'''
    array = np.linspace(-1., 0.8, 10).reshape((2, 1, 5, 1))
    bounds = interval_norm_bounds(array, symmetric=symmetric, dim=dim)
    assert np.allclose(expected_bounds, bounds)
