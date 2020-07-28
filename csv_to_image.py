import sys
import argparse
from pathlib import Path
import logging
from matplotlib.pylab import imsave
import numpy as np
import pandas as pd


MAX_DIFF = 5.0
# MAX_DIFF = 2000
COLOR_MAP = 'RdYlGn'
VALID_IMAGE_SUFFIXES = ('.png', '.jpg')

log = logging.getLogger(__name__)


def read_csv(file_path, chunk_size=None):
    log.info('Reading file {}'.format(file_path))
    df = pd.read_csv(file_path, header=None, dtype=np.float32, float_precision='high', chunksize=chunk_size)
    if chunk_size is None:
        log.info('Successful read {}X{} table'.format(df.columns.size, df.index.size))
    else:
        log.info('Reading file in chunks of {}'.format(chunk_size))
    return df

def convert_to_diff(df, minimum):
    df -= minimum
    # Remove negative values
    df[df < 0] = 0
    return df

def get_minimum(df):
    """
    Get the minimum value out of the positive values
    """
    return df[df > 0].min(numeric_only=True).min()

def df_to_image(df, image_path, minimum=None, maximum=None):
    minimum = minimum if minimum is not None else get_minimum(df)
    df = convert_to_diff(df, minimum=minimum)
    if maximum is not None:
        # Convert the maximum to "diff"
        maximum -= minimum
    else:
        maximum = df.max().max()
    log.info('Writing image to {} with maximum as {}'.format(image_path, maximum))
    imsave(image_path, df, cmap=COLOR_MAP, vmin=0, vmax=maximum)

def imagizer(file_path, image_path, chunk_size=None, minimum=None):
    """
    Turns a csv file into an image or multiple images
    """
    df = read_csv(file_path, chunk_size)
    if minimum is not None:
        log.info('Forced minimum value {}'.format(minimum))
    if chunk_size is None:
        df_to_image(df, image_path, minimum=minimum)
    else:
        if minimum is None:
            minimum = sys.maxsize
            cacl_min = True
        else:
            cacl_min = False
        maximum = 0
        for chunk in df:
            if cacl_min:
                minimum = min(minimum, get_minimum(chunk))
            maximum = max(maximum, chunk.max().max())
        log.info('Minimum value is {} and maximum is {}'.format(minimum, maximum))
        df = read_csv(file_path, chunk_size)
        for i, chunk in enumerate(df):
            image_chunk_path = image_path.parent / '{}_{:02}{}'.format(image_path.stem, i, image_path.suffix)
            df_to_image(chunk, image_chunk_path, minimum=minimum, maximum=maximum)

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
    imagizer(file_path, image_path, chunk_size=args.chunk_size, minimum=args.minimum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turns CSV files into images')
    parser.add_argument('file_path', type=Path, help='A path to a csv file')
    parser.add_argument('--image-path', '-o', type=Path, help='An optional path to the resulting image. If omitted will generate a path from the input path with a different suffix')
    parser.add_argument('--chunk-size', '-c', type=int, help='Separate the resulting image to chunks')
    parser.add_argument('--minimum', '-m', type=int, help='Set a hard minimum value and zero all values below it')
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
