# -*- coding: utf-8 -*-
"""
LIBS Generate

Generate LIBS emission spectra using the NIST database (https://physics.nist.gov/PhysRefData/ASD/LIBS/libs-form.html)
"""
import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

__author__ = 'Riccardo Finotello'
__email__ = 'riccardo.finotello@cea.fr'
__description__ = 'Generate LIBS emission spectra using the NIST database (https://physics.nist.gov/PhysRefData/ASD/LIBS/libs-form.html)'
__epilog__ = 'For bug reports and info: ' + __author__ + ' <' + __email__ + '>'


def main(args):

    # Seed the numpy random number generator for reproducibility and set the
    # matplotlib style
    rnd = np.random.RandomState(args.seed)
    mpl.style.use('ggplot')
    mpl.use('agg')

    # Read the input file
    df = pd.read_csv(args.input, sep=',', header=0, index_col=0)

    # Replace NaN by 0
    df.fillna(0, inplace=True)

    # Group and sum the columns by element:
    #
    # O I | O II | O III | Fe I | Fe II | Fe III | ...
    #
    # should become:
    #
    # O | Fe | ...
    df = df.groupby(df.columns.str.split(' ').str[0], axis=1).sum()

    # Remove the 'Sum' column, if present
    if 'Sum' in df.columns:
        df.drop('Sum', axis=1, inplace=True)

    # Plot the theoretical spectrum on the entire range. Then create a small
    # zoomed-in plot (500 nm - 900 nm) and insert it in
    # the main plot.
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(ax=ax, lw=1, legend=True)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('intensity (a.u.)')
    ax.set_title('Theoretical LIBS Spectrum')
    ax.set_xlim(200, 1000)
    ax.set_ylim(-0.1, 1.1 * df.max().max())
    ax.legend()
    ax.grid(True)

    axins = ax.inset_axes([0.35, 0.35, 0.5, 0.5])
    df.plot(ax=axins, lw=1, legend=False)
    axins.set_xlabel('wavelength (nm)')
    axins.set_ylabel('intensity (a.u.)')
    axins.set_xlim(500, 900)
    axins.set_ylim(-0.1, 1.1 * df.loc[500:900].max().max())
    axins.grid(True)

    ax.indicate_inset_zoom(axins)

    fig.savefig('libs_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # For each column, generate a spectrum using a Poisson distribution, whose
    # expectation value is the value of the respective row.
    spectra = np.zeros((*df.shape, args.num))
    for i in range(args.num):
        spectra[:, :, i] = np.abs(
            rnd.poisson(df.values) * rnd.normal(1, args.beta, df.shape)
            + rnd.normal(0, args.gamma, df.shape))

    # Convolve the spectra with a Gaussian kernel to simulate the instrument
    # response function.
    kernel = np.exp(-np.linspace(-5, 5, 100)**2 / args.gamma**2)
    kernel /= kernel.sum()
    spectra = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'),
                                  axis=0,
                                  arr=spectra)

    # Clip the values at 0.
    spectra = np.clip(spectra, 0, None)

    # Plot the average spectrum
    average = spectra.mean(axis=-1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, average, lw=1)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('intensity (a.u.)')
    ax.set_title('Average LIBS Spectrum (synthetic)')
    ax.set_xlim(200, 1000)
    ax.set_ylim(-0.1, 1.1 * average.max())
    ax.grid(True)

    axins = ax.inset_axes([0.35, 0.35, 0.5, 0.5])
    axins.plot(df.index, average, lw=1)
    axins.set_xlabel('wavelength (nm)')
    axins.set_ylabel('intensity (a.u.)')
    axins.set_xlim(500, 900)
    axins.set_ylim(-0.1,
                   1.1 * average[(df.index >= 500) & (df.index <= 900)].max())
    axins.grid(True)

    ax.indicate_inset_zoom(axins)

    fig.savefig('libs_spectrum_average.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create the output directory, if it does not exist
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # For each spectrum, create a DataFrame, add a 'Sum' column and save it to
    # a CSV file.
    for i in range(args.num):
        tmp = pd.DataFrame(spectra[:, :, i], index=df.index, columns=df.columns)
        tmp['Sum'] = tmp.sum(axis=1)
        tmp.to_csv(output / f'spectrum_{i:06d}.csv',
                   sep=',',
                   header=True,
                   index=True)

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__,
                                     epilog=__epilog__)
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='Input file')
    parser.add_argument('-n',
                        '--num',
                        type=int,
                        default=1,
                        help='Number of spectra to generate')
    parser.add_argument('-b',
                        '--beta',
                        type=float,
                        default=0.1,
                        help='Gaussian noise level')
    parser.add_argument('-c',
                        '--gamma',
                        type=float,
                        default=0.2,
                        help='Gaussian convolution width')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='output',
                        help='Output directory')
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)
