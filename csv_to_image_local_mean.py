import sys
import logging
import argparse
from pathlib import Path
import bisect
from functools import partial
import numpy as np
from csv_to_image import read_csv, init_logging


COLORS_GREATER = (100, 200, 300)
COLORS_LOWER = (-100, -200, -300)


log = logging.getLogger(__name__)


def bluntify_by_local(df, local_size, diffpoints):
    """
    
    :param df: The dataframe to work on
    :param local_size: A (index, column) tuple indicating the size of the local area
    :param diffpoints: A sorted by "diff" iterable of (diff, color_lower, color_greater) to compare to the mean.
                       Each of the df's cells' value in the range between two diffs from the mean will be changed
                       to the appropriate color (from the lower diffpoint).
                       All values closer than the smallest diffpoint will be changed to np.NAN
    """
    index, columns = df.shape
    for index_i in range(0, index, local_size[0]):
        for column_i in range(0, columns, local_size[1]):
            local_df = df.iloc[index_i:(index_i + local_size[0]), column_i:(column_i + local_size[1])]
            bluntify(local_df, diffpoints)

def bluntify(df, diffpoints):
    """
    
    :param df: The dataframe to work on
    :param diffpoints: A sorted by "diff" iterable of (diff, color_lower, color_greater) to compare to the mean.
                       Each of the df's cells' value in the range between two diffs from the mean will be changed
                       to the appropriate color (from the lower diffpoint).
                       All values closer than the smallest diffpoint will be changed to np.NAN
    """
    # Generate points by diffing the mean from each diffpoint
    mean = df.mean().mean()
    lower_points = []
    greater_points = []
    for d, color_lower, color_greater in diffpoints:
        # Keep the lists sorted
        lower_points.insert(0, (mean - d, color_lower))
        greater_points.append((mean + d, color_greater))
    # Apply in place for each cell
    df[:] = df.applymap(partial(find_bluntcolor, lower_points=lower_points, greater_points=greater_points))

def find_bluntcolor(num, lower_points, greater_points):
    """
    Find the closest point to num (including num itself)

    :param num: The number to compare to
    :param lower_points: A sorted by "value" iterable of (value, color) lower than the mean
    :param greater_points: A sorted by "value" iterable of (value, color) greater than the mean
    """
    lower_values, lower_colors = zip(*lower_points)
    i_lower = bisect.bisect_right(lower_values, num)
    if i_lower < len(lower_points):
        return lower_colors[i_lower]
    greater_values, greater_colors = zip(*greater_points)
    i_greater = bisect.bisect_left(greater_values, num)
    if i_greater == 0:
        return np.nan
    else:
        return greater_colors[i_greater - 1]

def verify_args(args):
    if not args.file_path.is_file():
        raise ValueError('file_path must be a path to file')

def main(args):
    init_logging(args.log_path)
    df = read_csv(args.file_path, chunksize=None)
    bluntify_by_local(df, local_size=(100, 100), diffpoints=(100, 200, 300))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turns CSV files into image by local mean')
    parser.add_argument('file_path', type=Path, help='A path to a csv file')
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
