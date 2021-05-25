'''Script to fit RGB heatmap images to a source color palette.'''
import click
import numpy as np
from PIL import Image

from zennit.image import CMAPS, palette


def gale_shapley(dist):
    '''Find a stable matching given a distance matrix.'''
    preference = np.argsort(dist, axis=1)
    proposed = np.zeros(dist.shape[0], dtype=int)
    loners = set(range(dist.shape[0]))
    guys = [-1] * dist.shape[0]
    gals = [-1] * dist.shape[1]
    while loners:
        loner = loners.pop()
        target = preference[loner, proposed[loner]]
        if gals[target] == -1:
            gals[target] = loner
            guys[loner] = target
        elif dist[gals[target], target] > dist[loner, target]:
            gals[target] = loner
            guys[loner] = target
            guys[gals[target]] = -1
            loners.add(gals[target])
        else:
            loners.add(loner)
        proposed[loner] += 1
    return guys


@click.command()
@click.argument('source-file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output-file', type=click.Path(writable=True, dir_okay=False))
@click.option('--strategy', type=click.Choice(['intensity', 'nearest', 'histogram']), default='intensity')
@click.option('--source-cmap', type=click.Choice(list(CMAPS)), default='bwr')
@click.option('--source-level', type=float, default=1.0)
@click.option('--invert/--no-invert', default=False)
@click.option('--cmap', type=click.Choice(list(CMAPS)), default='coldnhot')
@click.option('--level', type=float, default=1.0)
def main(source_file, output_file, strategy, source_cmap, source_level, invert, cmap, level):
    '''Fit an existing RGB heatmap image to a color palette.'''
    source = np.array(Image.open(source_file).convert('RGB'))
    matchpal = palette(source_cmap, source_level)

    if strategy == 'intensity':
        # matching based on the source image intensity/ brigthness
        values = source.astype(float).mean(2)
    elif strategy == 'nearest':
        # match by finding the neareast centroids in a source colormap
        dists = (np.abs(source[None].astype(float) - matchpal[:, None, None].astype(float))).sum(3)
        values = np.argmin(dists, axis=0)
    elif strategy == 'histogram':
        # match by finding a stable match between the color histogram of the source image and a source colormap
        source = np.concatenate([source, np.zeros_like(source[:, :, [0]])], axis=2).view(np.uint32)[..., 0]
        uniques, counts = np.unique(source, return_counts=True)
        uniques = uniques[np.argsort(counts)[-256:]]
        dist = (np.abs(uniques.view(np.uint8).reshape(-1, 1, 4)[..., :3] - matchpal[None])).sum(2)
        matches = np.array(gale_shapley(dist))

        ind_bin, ind_h, ind_w = np.nonzero(source[None] == uniques[:, None, None])
        values = np.zeros(source.shape[:2], dtype=np.uint8)
        values[ind_h, ind_w] = matches[ind_bin]

    values = values.clip(0, 255).astype(np.uint8)
    if invert:
        values = 255 - values

    img = Image.fromarray(values, mode='P')
    pal = palette(cmap, level)
    img.putpalette(pal)
    img.save(output_file)


if __name__ == '__main__':
    main()
