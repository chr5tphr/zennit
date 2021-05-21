'''Script to swap the palette of heatmap images.'''
import click
from PIL import Image

from zennit.image import CMAPS, palette


@click.command()
@click.argument('image-files', type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option('--cmap', type=click.Choice(list(CMAPS)), default='coldnhot')
@click.option('--level', type=float, default=1.0)
def main(image_files, cmap, level):
    '''Swap the palette of heatmap image files inline.'''
    for fname in image_files:
        img = Image.open(fname)
        img = img.convert('P')
        pal = palette(cmap, level)
        img.putpalette(pal)
        img.save(fname)


if __name__ == '__main__':
    main()
