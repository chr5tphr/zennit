'''Script to visually inspect color maps.'''
import click
import numpy as np
from PIL import Image

from zennit.image import CMAPS, palette


def semsstr(string):
    if isinstance(string, list):
        return string
    return [obj for obj in string.split(';') if obj]


@click.command()
@click.argument('output')
@click.option('--cmap', 'colormap_src', type=semsstr, default=list(CMAPS))
@click.option('--level', type=float, default=1.0)
def main(output, colormap_src, level):
    print('\n'.join(colormap_src))
    palettes = np.stack([palette(obj, level) for obj in colormap_src])
    arr = np.repeat(palettes, 32, 0)
    img = Image.fromarray(arr)
    img.save(output)


if __name__ == '__main__':
    main()
