'''Tests for ColorMap and CMSL.'''
from typing import NamedTuple
import pytest
import numpy as np

from zennit.cmap import ColorMap, LazyColorMapCache


class CMapExample(NamedTuple):
    '''Named tuple for example color maps used in tests.'''
    source: str
    nodes: list


CMAPS = [
    ('000,fff', [
        (0x00, (0x00, 0x00, 0x00)),
        (0xff, (0xff, 0xff, 0xff)),
    ]),
    ('fff,f00', [
        (0x00, (0xff, 0xff, 0xff)),
        (0xff, (0xff, 0x00, 0x00)),
    ]),
    ('fff,00f', [
        (0x00, (0xff, 0xff, 0xff)),
        (0xff, (0x00, 0x00, 0xff)),
    ]),
    ('000,f00,ff0,fff', [
        (0x00, (0x00, 0x00, 0x00)),
        (0x55, (0xff, 0x00, 0x00)),
        (0xaa, (0xff, 0xff, 0x00)),
        (0xff, (0xff, 0xff, 0xff)),
    ]),
    ('000,00f,0ff', [
        (0x00, (0x00, 0x00, 0x00)),
        (0x7f, (0x00, 0x00, 0xff)),
        (0xff, (0x00, 0xff, 0xff)),
    ]),
    ('0ff,00f,80:000,f00,ff0,fff', [
        (0x00, (0x00, 0xff, 0xff)),
        (0x40, (0x00, 0x00, 0xff)),
        (0x80, (0x00, 0x00, 0x00)),
        (0xaa, (0xff, 0x00, 0x00)),
        (0xd4, (0xff, 0xff, 0x00)),
        (0xff, (0xff, 0xff, 0xff)),
    ]),
    ('00f,80:fff,f00', [
        (0x00, (0x00, 0x00, 0xff)),
        (0x80, (0xff, 0xff, 0xff)),
        (0xff, (0xff, 0x00, 0x00)),
    ]),
    ('0055a4,80:fff,ef4135', [
        (0x00, (0x00, 0x55, 0xa4)),
        (0x80, (0xff, 0xff, 0xff)),
        (0xff, (0xef, 0x41, 0x35)),
    ]),
    ('0000d0,80:d0d0d0,d00000', [
        (0x00, (0x00, 0x00, 0xd0)),
        (0x80, (0xd0, 0xd0, 0xd0)),
        (0xff, (0xd0, 0x00, 0x00)),
    ]),
    ('00d0d0,80:d0d0d0,d000d0', [
        (0x00, (0x00, 0xd0, 0xd0)),
        (0x80, (0xd0, 0xd0, 0xd0)),
        (0xff, (0xd0, 0x00, 0xd0)),
    ]),
    ('00d000,80:d0d0d0,d000d0', [
        (0x00, (0x00, 0xd0, 0x00)),
        (0x80, (0xd0, 0xd0, 0xd0)),
        (0xff, (0xd0, 0x00, 0xd0)),
    ]),
    ('7:000, 9:ffffff', [
        (0x00, (0x00, 0x00, 0x00)),
        (0x77, (0x00, 0x00, 0x00)),
        (0x99, (0xff, 0xff, 0xff)),
        (0xff, (0xff, 0xff, 0xff)),
    ]),
]


def interpolate(x, nodes):
    '''Interpolate from example color map nodes.'''
    xp_addr = np.array([node[0] for node in nodes], dtype=np.float64)
    fp_rgb = np.array([node[1] for node in nodes], dtype=np.float64).T
    return np.stack([np.interp(x, xp_addr, fp) for fp in fp_rgb], axis=-1).round(12).clip(0., 255.).astype(np.uint8)


@pytest.fixture(scope='session', params=CMAPS)
def cmap_example(request):
    '''Example color map fixture.'''
    return CMapExample(*request.param)


@pytest.mark.parametrize('source_code', [
    'this', 'fff', ',,,', '111:111:111', 'fffff,fffff', 'f,f', 'fffffffff', 'ff:', 'ff:fff,00:000'
])
def test_color_map_wrong_syntax(source_code):
    '''Test whether different kinds of syntax errors cause a RuntimeError.'''
    with pytest.raises(RuntimeError):
        ColorMap(source_code)


def test_color_map_nodes_call(cmap_example):
    '''Test if the color map nodes have the specified color when calling a ColorMap instance.'''
    cmap = ColorMap(cmap_example.source)
    input_addr = np.array([node[0] for node in cmap_example.nodes], dtype=np.float64)[None]
    expected_rgb = np.array([node[1] for node in cmap_example.nodes], dtype=np.uint8)[None]
    cmap_rgb = (cmap(input_addr / 255.) * 255.).round(12).clip(0., 255.).astype(np.uint8)
    assert np.allclose(expected_rgb, cmap_rgb)


def test_color_map_nodes_palette(cmap_example):
    '''Test if the color map nodes have the specified color when using ColorMap.palette.'''
    cmap = ColorMap(cmap_example.source)
    input_addr = [node[0] for node in cmap_example.nodes]
    expected_rgb = np.array([node[1] for node in cmap_example.nodes], dtype=np.uint8)[None]
    palette = cmap.palette(level=1.)
    cmap_rgb = palette[input_addr]
    assert np.allclose(expected_rgb, cmap_rgb)


def test_color_map_full_call(cmap_example):
    '''Test if the color map nodes have correctly interpolated colors when calling a ColorMap instance.'''
    cmap = ColorMap(cmap_example.source)
    input_addr = np.arange(256, dtype=np.uint8)
    expected_rgb = interpolate(input_addr, cmap_example.nodes)
    cmap_rgb = (cmap(input_addr / 255.) * 255.).round(12).clip(0., 255.).astype(np.uint8)
    assert np.allclose(expected_rgb, cmap_rgb)


def test_color_map_full_palette(cmap_example):
    '''Test if the color map nodes have correctly interpolated colors when using ColorMap.palette.'''
    input_addr = np.arange(256, dtype=np.uint8)
    expected_palette = interpolate(input_addr, cmap_example.nodes)
    cmap = ColorMap(cmap_example.source)
    cmap_palette = cmap.palette(level=1.0)
    assert np.allclose(expected_palette, cmap_palette)


def test_color_map_reassign_source_palette(cmap_example):
    '''Test if calling a ColorMap instance for which the source was changed produces correctly interpolated colors.'''
    cmap = ColorMap('fff,fff')
    cmap.source = cmap_example.source

    input_addr = np.arange(256, dtype=np.uint8)
    expected_palette = interpolate(input_addr, cmap_example.nodes)
    cmap_palette = cmap.palette(level=1.0)
    assert np.allclose(expected_palette, cmap_palette)


def test_color_map_source_property(cmap_example):
    '''Test if the source property of a color map is equal to the specified source code.'''
    cmap = ColorMap(cmap_example.source)
    assert cmap.source == cmap_example.source, 'Mismatching source!'


@pytest.fixture(scope='function')
def lazy_cmap_cache():
    '''Single fixture for a LazyColorMapCache'''
    return LazyColorMapCache({
        'gray': '000,fff',
        'red': '100,f00',
    })


class TestLazyColorMapCache:
    '''Tests for LazyColorMapCache.'''
    @staticmethod
    def test_missing(lazy_cmap_cache):
        '''Test whether accessing an unknown key causes a KeyError.'''
        with pytest.raises(KeyError):
            _ = lazy_cmap_cache['no such cmap']

    @staticmethod
    def test_get_item_uncompiled(lazy_cmap_cache):
        '''Test whether accessing an uncompiled entry compiles and returns the correct ColorMap.'''
        cmap = lazy_cmap_cache['red']
        assert isinstance(cmap, ColorMap)
        assert cmap.source == '100,f00'

    @staticmethod
    def test_get_item_cached(lazy_cmap_cache):
        '''Test whether accessing a previously compiled and cached entry returns the same ColorMap.'''
        cmaps = [
            lazy_cmap_cache['red'],
            lazy_cmap_cache['red'],
        ]
        assert cmaps[0] is cmaps[1]

    @staticmethod
    def test_set_item_existing(lazy_cmap_cache):
        '''Test whether setting an already existing, uncompiled entry and accessing it returns the correct ColorMap.'''
        lazy_cmap_cache['red'] = 'fff,f00'
        assert lazy_cmap_cache['red'].source == 'fff,f00'

    @staticmethod
    def test_set_item_new(lazy_cmap_cache):
        '''Test whether setting a new entry and accessing it returns the correct ColorMap.'''
        lazy_cmap_cache['blue'] = 'fff,00f'
        assert lazy_cmap_cache['blue'].source == 'fff,00f'

    @staticmethod
    def test_set_item_compiled(lazy_cmap_cache):
        '''Test whether setting an already existing, compiled entry and accessing it returns the same, modified
        ColorMap instance.
        '''
        original_cmap = lazy_cmap_cache['red']
        lazy_cmap_cache['red'] = 'fff,f00'
        assert lazy_cmap_cache['red'].source == 'fff,f00'
        assert original_cmap is lazy_cmap_cache['red']

    @staticmethod
    def test_del_item_uncompiled(lazy_cmap_cache):
        '''Test whether deleting an uncompiled entry correctly removes the entry.'''
        del lazy_cmap_cache['red']
        assert 'red' not in lazy_cmap_cache

    @staticmethod
    def test_del_item_compiled(lazy_cmap_cache):
        '''Test whether deleting a compiled entry correctly removes the entry.'''
        _ = lazy_cmap_cache['red']
        del lazy_cmap_cache['red']
        assert 'red' not in lazy_cmap_cache

    @staticmethod
    def test_iter(lazy_cmap_cache):
        '''Test whether iterating a LazyColorMapCache returns its keys.'''
        assert (list(lazy_cmap_cache) == ['gray', 'red'])

    @staticmethod
    def test_len(lazy_cmap_cache):
        '''Test whether calling len on a LazyColorMapCache returns the correct length.'''
        assert len(lazy_cmap_cache) == 2
