import sys
import argparse
from pathlib import Path
import logging
from collections import abc
from matplotlib.pylab import imsave
import numpy as np
import pandas as pd


# COLOR_MAP = 'RdYlGn'
# COLOR_MAP = 'YlGnBu'
COLOR_MAP = 'BuPu'
VALID_IMAGE_SUFFIXES = ('.png', '.jpg')
DTYPE = np.float32

log = logging.getLogger(__name__)


class ResetableTextFileReader(abc.Iterator):
    def __init__(self, text_file_reader, reset_func, *func_args, **func_kwargs):
        self._text_file_reader = text_file_reader
        self._reset_func = reset_func
        self._func_args = func_args
        self._func_kwargs = func_kwargs

    def __next__(self):
        return next(self._text_file_reader)

    def reset(self):
        self._text_file_reader.close()
        self._text_file_reader = self._reset_func(*self._func_args, **self._func_kwargs)

    def close(self):
        return self._text_file_reader.close()

    def read(self, nrows=None):
        return self._text_file_reader.read(nrows=nrows)

    def get_chunk(self, size=None):
        return self._text_file_reader.get_chunk(size=size)


def read_csv(file_path, chunksize=None):
    log.info('Reading file {}'.format(file_path))
    read_csv_kwargs = {'dtype': DTYPE, 'index_col': 0, 'engine': 'python', 'chunksize': chunksize}
    df = pd.read_csv(file_path, **read_csv_kwargs)
    if chunksize is None:
        log.info('Successful read {}X{} table'.format(df.columns.size, df.index.size))
    else:
        log.info('Reading file in chunks of {}'.format(chunksize))
        df = ResetableTextFileReader(df, pd.read_csv, file_path, **read_csv_kwargs)
    return df

def get_minimum(df):
    """
    Get the minimum value out of the positive values
    """
    return df[df > 0].min(numeric_only=True).min()

# def impose_edge_values_limit(df, minimum, maximum):
#     """
#     Imposes a limit on df's values by input minimum and maximum.
#     All negative values or values lower than minimum become NaN (transparent in the final image)
#     All values larger than maximum become become NaN (transparent in the final image)
#     """
#     is_success = True
#     real_minimum = get_minimum(df)
#     if minimum:
#         # Reset all values below input minimum
#         df[df < minimum] = np.nan
#         if real_minimum < minimum:
#             log.warning('Input minimum {} is larger than the real minimum {}'.format(minimum, real_minimum))
#             is_success = False
#     else:
#         # Reset negative values to 0
#         df[df < 0] = np.nan
#         minimum = real_minimum
#     real_maximum = df.max().max()
#     if maximum:
#         # Reset all values above input maximum
#         df[df > maximum] = np.nan
#         if real_maximum > maximum:
#             log.warning("Input maximum {} is lower than the real maximum {}".format(maximum, real_maximum))
#             is_success = False
#     else:
#         maximum = real_maximum
#     return is_success, minimum, maximum

def impose_edge_values_limit(df, minimum, maximum):
    """
    Imposes a limit on df's values by input minimum and maximum.
    All negative values or values lower than minimum become NaN (transparent in the final image)
    All values larger than maximum become become NaN (transparent in the final image)
    """
    below_min_df, above_max_df = None, None
    real_minimum = get_minimum(df)
    if minimum:
        if real_minimum < minimum:
            log.warning('Input minimum {} is larger than the real minimum {}'.format(minimum, real_minimum))
            # Reset all values below input minimum
            # below_min_df = df[(df < minimum) & (df > 0)]
            below_min_df = df[df < minimum]
            df[df < minimum] = np.nan
    else:
        # Reset negative values to 0
        df[df < 0] = np.nan
        minimum = real_minimum
    real_maximum = df.max().max()
    if maximum:
        # Reset all values above input maximum
        if real_maximum > maximum:
            log.warning("Input maximum {} is lower than the real maximum {}".format(maximum, real_maximum))
            above_max_df = df[df > maximum]
            df[df > maximum] = np.nan
    else:
        maximum = real_maximum
    return below_min_df, above_max_df, minimum, maximum

def df_to_image(df, image_path, minimum=None, maximum=None):
    below_min_df, above_max_df, minimum, maximum = impose_edge_values_limit(df, minimum, maximum)
    log.info('Writing image to {} while splitting the color range between ({}, {})'.format(image_path, minimum, maximum))
    imsave(image_path, df, cmap=COLOR_MAP, vmin=minimum, vmax=maximum)
    if below_min_df is not None or above_max_df is not None:
        if below_min_df is not None and above_max_df is not None:
            below_min_df.add(above_max_df, fill_value=0)
            diff_df = below_min_df
        elif below_min_df is not None:
            diff_df = below_min_df
        else:
            diff_df = above_max_df
        del above_max_df, below_min_df
        invalid_image_path = image_path.parent / (image_path.stem + '_invalid' + image_path.suffix)
        log.warning('Saving invalid values to {}'.format(invalid_image_path))
        import pdb; pdb.set_trace()
        imsave(invalid_image_path, diff_df, cmap='Reds')
        return False
    return True

def get_chunks_mean(df_chunks):
    if isinstance(df_chunks, pd.DataFrame):
        # df_chunks is not separated into chunks
        return df_chunks.mean().mean()
    mean_generator = (chunk.mean().mean() for chunk in df_chunks)
    mean = pd.Series(mean_generator).mean()
    df_chunks.reset()
    return mean

def get_chunks_minmax(df_chunks, minimum=None, maximum=None):
    """
    Calculated the min/max values if wasn't given them.
    """
    if minimum is not None and maximum is not None:
        return minimum, maximum
    if isinstance(df_chunks, pd.DataFrame):
        # df_chunks is not separated into chunks
        return minimum, maximum
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
    df_chunks.reset()
    return minimum, maximum


def calculate_minmax_values(df, minimum=None, maximum=None, mean_lower_diff=None, mean_upper_diff=None):
    msg = ''
    if maximum or maximum:
        minimum, maximum = get_chunks_minmax(df, minimum=minimum, maximum=maximum)
    elif (mean_lower_diff or mean_upper_diff):
        mean = get_chunks_mean(df)
        msg += 'The mean is {}. '.format(mean)
        if mean_lower_diff:
            minimum = mean - mean_lower_diff
        if mean_upper_diff:
            maximum = mean + mean_upper_diff
    if minimum:
        msg += "Minimum is {}. ".format(minimum)
    if maximum:
        msg += "Maximum is {}.".format(maximum)
    if msg:
        log.info(msg)
    return minimum, maximum

def imagizer(file_path, image_path, chunksize=None, minimum=None, maximum=None, mean_lower_diff=None, mean_upper_diff=None):
    """
    Turns a csv file into an image or multiple images
    """
    df = read_csv(file_path, chunksize)
    minimum, maximum = calculate_minmax_values(df, minimum=minimum, maximum=maximum,
                                               mean_lower_diff=mean_lower_diff, mean_upper_diff=mean_upper_diff)
    if chunksize is None:
        return df_to_image(df, image_path, minimum=minimum, maximum=maximum)
    else:
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
    if (args.maximum or args.maximum) and (args.mean_lower_diff or args.mean_upper_diff):
        raise ValueError("Must not use both hard min/max limitations and mead diff limitations")

def init_logging(log_path=None):
    logging.basicConfig(filename=log_path, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel('INFO')

def main(args):
    init_logging(args.log_path)
    file_path = args.file_path
    image_path = args.image_path if args.image_path else file_path.parent / (file_path.stem + '.png')
    success = imagizer(file_path, image_path, chunksize=args.chunksize, 
        minimum=args.minimum, maximum=args.maximum, mean_lower_diff=args.mean_lower_diff, mean_upper_diff=args.mean_upper_diff)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turns CSV files into images')
    parser.add_argument('file_path', type=Path, help='A path to a csv file')
    parser.add_argument('--image-path', '-o', type=Path, help='An optional path to the resulting image. If omitted will generate a path from the input path with a different suffix')
    parser.add_argument('--chunksize', '-c', type=int, help='Separate the resulting image to chunks')
    parser.add_argument('--minimum', '-m', type=float, help='Set a hard minimum value')
    parser.add_argument('--maximum', '-x', type=float, help='Set a hard maximum value')
    parser.add_argument('--mean-lower-diff', '-ml', type=float, help='Subtract this value from the mean (average) and set it as a minimum value')
    parser.add_argument('--mean-upper-diff', '-mu', type=float, help='Add this value to the mean (average) and set it as a maximum value')
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
