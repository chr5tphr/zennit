'''Functionality to convert arrays to images'''
import numpy as np
from PIL import Image

import matplotlib.cm


CMAPS = {}


def register_cmap(name):
    '''Decorator to register a color map.'''
    def wrapped(func):
        '''Wrapped function to register a color map with name `name`.'''
        CMAPS[name] = func
        return func

    return wrapped


@register_cmap('gray')
def gray(x):
    '''Color map from black to white.'''
    return np.stack([x] * 3, axis=-1).clip(0., 1.)


@register_cmap('wh_rd')
def wh_rd(x):
    '''Color map from white to red.'''
    return np.stack([0. * x + 1., 1. - x, 1. - x], axis=-1).clip(0., 1.)


@register_cmap('wh_bu')
def wh_bu(x):
    '''Color map from white to blue.'''
    return np.stack([1. - x, 1. - x, 0 * x + 1.], axis=-1).clip(0., 1.)


@register_cmap('wh_gn')
def wh_gn(x):
    '''Color map from white to green.'''
    return np.stack([1. - x, 0 * x + 1., 1. - 0.5 * x], axis=-1).clip(0., 1.)


@register_cmap('wh_mg')
def wh_mg(x):
    '''Color map from white to magenta.'''
    return np.stack([0 * x + 1., 1 - x, 0 * x + 1.], axis=-1).clip(0., 1.)


@register_cmap('wh_cy')
def wh_cy(x):
    '''Color map from white to cyan.'''
    return np.stack([1. - 230/255*x, 1. - 40/255*x, 1. - 55/255*x], axis=-1).clip(0., 1.)


@register_cmap('wh_bu')
def wh_bu(x):
    '''Color map from white to blue.'''
    return np.stack([1. - x, 1. - x, 0 * x + 1.], axis=-1).clip(0., 1.)


@register_cmap('bk_mg')
def bk_mg(x):
    '''Color map from black to magenta.'''
    return np.stack([1. * x, 0. * x, 1. * x], axis=-1).clip(0., 1.)


@register_cmap('bk_yl')
def bk_yl(x):
    '''Color map from black to yellow.'''
    return np.stack([1. * x, 1. * x, 0. * x], axis=-1).clip(0., 1.)


@register_cmap('bk_or')
def bk_or(x):
    '''Color map from black to orange.'''
    return np.stack([2. * x, 2. * x - 1, 0. * x], axis=-1).clip(0., 1.)


@register_cmap('bk_cy')
def bk_cy(x):
    '''Color map from black to cyan.'''
    return np.stack([0. * x, 1. * x, 1. * x], axis=-1).clip(0., 1.)


@register_cmap('hot')
def hot(x):
    '''Color map from black to red to yellow to white.'''
    return np.stack([x * 3., x * 3. - 1, x * 3 - 2], axis=-1).clip(0., 1.)


@register_cmap('cold')
def cold(x):
    '''Color map from black to blue to cyan.'''
    return np.stack([0. * x, x * 2. - 1., x * 2], axis=-1).clip(0., 1.)


@register_cmap('bk_pu_mg')
def bk_pu_mg(x):
    '''Color map from black to purple to magenta.'''
    return np.stack([x * 2. - 1., 0. * x, x * 2], axis=-1).clip(0., 1.)


@register_cmap('bk_gn_cy')
def bk_gn_cy(x):
    '''Color map from black to green to cyan.'''
    return np.stack([0. * x, x * 2, x * 2. - 1], axis=-1).clip(0., 1.)


@register_cmap('coldnhot')
def coldnhot(x):
    '''Combination of color maps cold (reversed) and hot.
    Colors range from cyan to blue to black to red to yellow to white.
    '''
    return hot((2 * x - 1.).clip(0., 1.)) + cold(-(2 * x - 1.).clip(-1., 0.))


@register_cmap('cy_gn_bk_pu_mg')
def cy_gn_bk_pu_mg(x):
    '''Combination of color maps bk_gn_cy (reversed) and bk_pu_mg
    Colors range from cyan to green to black to purple to magenta.
    '''
    return bk_pu_mg((2 * x - 1.).clip(0., 1.)) + bk_gn_cy(-(2 * x - 1.).clip(-1., 0.))


@register_cmap('mg_pu_bk_gr_cy')
def mg_pu_bk_gr_cy(x):
    '''Combination of color maps bk_pu_mg (reversed) and bk_gn_cy.
    Colors range from magenta to purple to black to green to cyan.
    '''
    return bk_gn_cy((2 * x - 1.).clip(0., 1.)) + bk_pu_mg(-(2 * x - 1.).clip(-1., 0.))


@register_cmap('cy_bk_mg')
def cy_bk_mg(x):
    '''Combination of color maps bk_cy (reversed) and bk_mg.
    Colors range from cyan to black to magenta.
    '''
    return bk_mg((2 * x - 1.).clip(0., 1.)) + bk_cy(-(2 * x - 1.).clip(-1., 0.))


@register_cmap('mg_bk_cy')
def cy_bk_mg(x):
    '''Combination of color maps bk_mg (reversed) and bk_cy.
    Colors range from magenta to black to cyan.
    '''
    return bk_cy((2 * x - 1.).clip(0., 1.)) + bk_mg(-(2 * x - 1.).clip(-1., 0.))


@register_cmap('yl_bk_mg')
def yl_bk_mg(x):
    '''Combination of color maps bk_yl (reversed) and bk_mg.
    Colors range from yellow to black to magenta.
    '''
    return bk_mg((2 * x - 1.).clip(0., 1.)) + bk_yl(-(2 * x - 1.).clip(-1., 0.))


@register_cmap('bu_wh_rd')
def bu_wh_rd(x):
    '''Combination of color maps wh_bu (reveresed) and wh_rd.
    Colors range from blue to white to red.
    '''
    return wh_rd((2 * x - 1.).clip(0., 1.)) + wh_bu(-(2 * x - 1.).clip(-1., 0.)) - 1.


@register_cmap('france')
def france(x):
    '''Combination of color maps wh_bu (reversed) and wh_rd (factor 0.85*0.96).
    Colors range from blue to white to red.
    '''
    return 0.85 * (wh_rd((2 * x - 1.).clip(0., 1.)) + wh_bu(-(2 * x - 1.).clip(-1., 0.)) - 1.) *0.96


@register_cmap('coleus')
def coleus(x):
    '''Combination of color maps wh_gn (reversed) and wh_mg (factor 0.85*0.96).
    Colors range from magenta to grey to green.
    '''
    return 0.85 * (wh_mg((2 * x - 1.).clip(0., 1.)) + wh_gn(-(2 * x - 1.).clip(-1., 0.)) - 1.) *0.96


@register_cmap('coolio')
def coolio(x):
    '''Combination of color maps wh_cy (reversed) and wh_mg (factor 0.85*0.96).
    Colors range from cyan to grey to magenta.
    '''
    return 0.85 * (wh_mg((2 * x - 1.).clip(0., 1.)) + wh_cy(-(2 * x - 1.).clip(-1., 0.)) - 1.) *0.96


@register_cmap('seismic085')
def seismic085(x):
    '''
    Seismic, but sucks less.
    '''
    tmp = matplotlib.cm.get_cmap('seismic')(x)
    tmp = tmp[..., 0:3]
    return tmp*0.85


def palette(cmap='bu_wh_rd', level=1.0):
    '''Create a 8-bit palette.

    Parameters
    ----------
    cmap: str
        String to describe the color map used to create the palette.
    level: float
        The level of the color map. 1.0 is default. Values below zero reduce the color range, with only a single color
        used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly steep
        transition at the center of the image.

    Returns
    -------
    obj:`numpy.ndarray`
        The palette described by an unsigned 8-bit numpy array with 256 entries.
    '''
    x = np.linspace(-1., 1., 256) * level
    x = ((x + 1.) / 2).clip(0., 1.)
    x = CMAPS[cmap](x)
    x = (x * 255.).clip(0., 255.).astype(np.uint8)
    return x


def imgify(obj, vmin=None, vmax=None, cmap='bu_wh_rd', level=1.0):
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
    cmap: str
        Color-map described by a string. Possible values are in the CMAPS dict. The color map will only be applied for
        arrays with only a single color channel. The color will be specified as a palette in the PIL Image.
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


def imsave(fp, obj, vmin=None, vmax=None, cmap='bu_wh_rd', level=1.0, grid=False, format=None, writer_params=None):
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
    cmap: str
        Color-map described by a string. Possible values are in the CMAPS dict. The color map will only be applied for
        arrays with only a single color channel. The color will be specified as a palette in the PIL Image.
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
