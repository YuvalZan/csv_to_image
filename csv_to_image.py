import sys
import argparse
from pathlib import Path
import logging
from matplotlib.pylab import imsave
import numpy as np
import pandas as pd


COLOR_MAP = 'RdYlGn'
VALID_IMAGE_SUFFIXES = ('.png', '.jpg')
DTYPE = np.float32

log = logging.getLogger(__name__)


def read_csv(file_path, chunk_size=None):
    log.info('Reading file {}'.format(file_path))
    df = pd.read_csv(file_path, dtype=DTYPE, index_col=0, engine='python', chunksize=chunk_size)
    if chunk_size is None:
        log.info('Successful read {}X{} table'.format(df.columns.size, df.index.size))
    else:
        log.info('Reading file in chunks of {}'.format(chunk_size))
    return df

def get_minimum(df):
    """
    Get the minimum value out of the positive values
    """
    return df[df > 0].min(numeric_only=True).min()

def impose_edge_values_limit(df, minimum, maximum):
    """
    Imposes a limit on df's values by input minimum and maximum.
    All negative values or values lower than minimum become NaN (transparent in the final image)
    All values larger than maximum become maximum
    """
    is_success = True
    real_minimum = get_minimum(df)
    if minimum:
        # Reset all values below input minimum
        df[df < minimum] = np.nan
        if real_minimum < minimum:
            log.warning('Input minimum {} is larger than the real minimum {}'.format(minimum, real_minimum))
            is_success = False
    else:
        # Reset negative values to 0
        df[df < 0] = np.nan
        minimum = real_minimum
    real_maximum = df.max().max()
    if maximum:
        # Reset all values above input maximum to itself
        df[df > maximum] = maximum
        if real_maximum > maximum:
            log.warning("Input maximum {} is lower than the real maximum {}".format(maximum, real_maximum))
            is_success = False
    else:
        maximum = real_maximum
    return is_success, minimum, maximum

def df_to_image(df, image_path, minimum=None, maximum=None):
    is_success, minimum, maximum = impose_edge_values_limit(df, minimum, maximum)
    log.info('Writing image to {} while splitting the color range between ({}, {})'.format(image_path, minimum, maximum))
    imsave(image_path, df, cmap=COLOR_MAP, vmin=minimum, vmax=maximum)
    return is_success

def get_chunks_minmax(df_chunks, minimum=None, maximum=None):
    """
    Calculated the min/max values if wasn't given them.
    Will "use-up" the df_chunks iterator
    """
    cacl_min, cacl_max = False, False
    if minimum is None:
        minimum = sys.maxsize
        cacl_min = True
    if maximum is None:
        maximum = 0
        cacl_max = True
    for chunk in df_chunks:
        if cacl_min:
            minimum = min(minimum, get_minimum(chunk))
        if cacl_max:
            maximum = max(maximum, chunk.max().max())
    return minimum, maximum

def imagizer(file_path, image_path, chunk_size=None, minimum=None, maximum=None):
    """
    Turns a csv file into an image or multiple images
    """
    df = read_csv(file_path, chunk_size)
    if minimum is not None:
        log.info('Forced minimum value {}'.format(minimum))
    if maximum is not None:
        log.info('Forced maximum value {}'.format(maximum))
    if chunk_size is None:
        return df_to_image(df, image_path, minimum=minimum, maximum=maximum)
    else:
        if minimum is not None or maximum is not None:
            minimum, maximum = get_chunks_minmax(df, minimum=minimum, maximum=maximum)
            log.info('Calculated minimum and maximum are {}, {} respectively'.format(minimum, maximum))
            df = read_csv(file_path, chunk_size)
        else:
            log.info('input minimum and maximum are {}, {} respectively'.format(minimum, maximum))
        success = True
        for i, chunk in enumerate(df):
            image_chunk_path = image_path.parent / '{}_{:02}{}'.format(image_path.stem, i, image_path.suffix)
            success_chunk = df_to_image(chunk, image_chunk_path, minimum=minimum, maximum=maximum)
            success = False if (not success or not success_chunk) else True
        return success

def verify_args(args):
    if not args.file_path.is_file():
        raise ValueError('file_path must be a path to file')
    if args.image_path:
        if not args.image_path.parent.is_dir():
            raise ValueError("Invalid image_path! Must be a valid file path")
    if args.log_path:
        if not args.log_path.parent.is_dir():
            raise ValueError("Invalid log_path! Must be a valid file path")
    if args.image_path and args.image_path.suffix not in VALID_IMAGE_SUFFIXES:
        raise ValueError("Invalid image_path! Must be in {}".format(VALID_IMAGE_SUFFIXES))

def init_logging(log_path=None):
    logging.basicConfig(filename=log_path, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel('INFO')

def main(args):
    init_logging(args.log_path)
    file_path = args.file_path
    image_path = args.image_path if args.image_path else file_path.parent / (file_path.stem + '.png')
    success = imagizer(file_path, image_path, chunk_size=args.chunk_size, minimum=args.minimum, maximum=args.maximum)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turns CSV files into images')
    parser.add_argument('file_path', type=Path, help='A path to a csv file')
    parser.add_argument('--image-path', '-o', type=Path, help='An optional path to the resulting image. If omitted will generate a path from the input path with a different suffix')
    parser.add_argument('--chunk-size', '-c', type=int, help='Separate the resulting image to chunks')
    parser.add_argument('--minimum', '-m', type=int, help='Set a hard minimum value and zero all values below it')
    parser.add_argument('--maximum', '-x', type=int, help='Set a hard maximum value and flatten all values above it to itself')
    parser.add_argument('--log-path', '-l', type=Path, help='Save the log to a file instead of stdout')
    try:
        args = parser.parse_args()
        verify_args(args)
    except Exception:
        log.exception('Argument validation failed')
        sys.exit(2)
    try:
        main(args)
    except Exception:
        log.exception('An exception was raised:')
        sys.exit(2)
