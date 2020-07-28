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

log = logging.getLogger(__name__)


def df_to_image(df, image_path, **kwargs):
    log.info('Writing image to {}'.format(image_path))
    imsave(image_path, df, cmap=COLOR_MAP, **kwargs)

def read_csv(file_path, chunk_size=None):
    log.info('Reading file {}'.format(file_path))
    df = pd.read_csv(file_path, header=None, dtype=np.float32, float_precision='high', chunksize=chunk_size)
    if chunk_size is None:
        log.info('Successful read {}X{} table'.format(df.columns.size, df.index.size))
    else:
        log.info('Reading in chunks of {}'.format(chunk_size))
    return df

def convert_to_diff(df):
    # Remove negative values
    df[df < 0] = 0
    minimum = df[df > 0].min(numeric_only=True).min()
    maximum = df.max().max()
    log.info('The minimum is {} and the maximum is {}'.format(minimum, maximum))
    # # Make the minimum be 0
    # df -= minimum
    # df[df < 0] = 0
    # maximum -= minimum
    if maximum > MAX_DIFF:
        log.warning('Maximum diff {} is bigger than the limit {}'.format(maximum, MAX_DIFF))
        # sys.exit(1)
    return df

def normalize_df(df):
    mean = df.mean()
    log.info('Normalizing table by mean {}'.format(mean.mean()))
    return (df - mean / df.std())

def raw_df_to_image(df, image_path, is_normalize=False):
    df = convert_to_diff(df)    
    if is_normalize:
        df = normalize_df(df)
    # df_to_image(df, image_path, vmin=0, vmax=MAX_DIFF)
    df_to_image(df, image_path)

def imagizer(file_path, image_path, is_normalize=False, chunk_size=None):
    df = read_csv(file_path, chunk_size)
    if chunk_size is None:
        raw_df_to_image(df, image_path, args.normalize)
    else:
        for i, chunk in enumerate(df):
            image_chunk_path = image_path.parent / '{}_{:02}{}'.format(image_path.stem, i, image_path.suffix)
            raw_df_to_image(chunk, image_chunk_path, is_normalize)

def verify_args(args):
    if not args.file_path.is_file():
        raise ValueError('file_path must be a path to file')
    if args.image_path:
        if not args.image_path.parent.is_dir():
            raise ValueError("Invalid image_path! Must be a valid file path")
    if args.log_path:
        if not args.log_path.parent.is_dir():
            raise ValueError("Invalid log_path! Must be a valid file path")

def init_logging(log_path=None):
    logging.basicConfig(filename=log_path, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel('INFO')

def main(args):
    init_logging(args.log_path)
    file_path = args.file_path
    image_path = args.image_path if args.image_path else file_path.parent / (file_path.stem + '.png')
    imagizer(file_path, image_path, is_normalize=args.normalize, chunk_size=args.chunk_size)
    # df = read_csv(file_path)
    # raw_df_to_image(df, image_path, args.normalize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turns CSV files into images')
    parser.add_argument('file_path', type=Path, help='A path to a csv file')
    parser.add_argument('--image-path', '-m', type=Path, help='An optional path to the resulting image. If omitted will generate a path from the input path with a different suffix')
    parser.add_argument('--chunk-size', '-c', type=int, help='Separate the resulting image to chunks')
    parser.add_argument('--normalize', '-n',action='store_true', help='Normalize by mean before converting to image')
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
