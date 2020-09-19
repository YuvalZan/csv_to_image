import sys
import logging
import argparse
from pathlib import Path
import bisect
from functools import partial
import multiprocessing
import math
import numpy as np
import pandas as pd
from matplotlib.pylab import imsave
from csv_to_image import read_csv, init_logging, VALID_IMAGE_SUFFIXES



COLOR_MAP = 'Paired'
PAIRED_MIN, PAIRED_MAX = 1, 12
PAIRED_LIGHT_BLUE = 1
PAIRED_DARK_BLUE = 2
PAIRED_LIGHT_GREEN = 3
PAIRED_DARK_GREEN = 4
PAIRED_LIGHT_RED = 5
PAIRED_DARK_RED = 6
PAIRED_LIGHT_ORANGE = 7
PAIRED_DARK_ORANGE = 8
PAIRED_LIGHT_PURPLE = 9
PAIRED_DARK_PURPLE = 10
PAIRED_YELLOW = 11
PAIRED_BROWN = 12

COLORS_GREATER = (PAIRED_LIGHT_BLUE, PAIRED_DARK_BLUE, PAIRED_LIGHT_GREEN, PAIRED_DARK_GREEN)
COLORS_LOWER = (PAIRED_LIGHT_ORANGE, PAIRED_DARK_ORANGE, PAIRED_LIGHT_RED, PAIRED_DARK_RED)
MAX_DIFFPOINTS = len(COLORS_LOWER)


log = logging.getLogger('csv_to_image')



# def parallelize(df, func):
#     num_cores = multiprocessing.cpu_count()
#     df_split = np.array_split(df, num_cores * 2)
#     pool = multiprocessing.Pool(num_cores * 2)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df

# def parallelize_func(iterable, func):
#     num_cores = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(num_cores * 2)
#     results = pool.map(func, iterable)
#     pool.close()
#     pool.join()
#     return results

def create_mean_thumbnail(df, ratio):
    """
    Creates a smaller mean_df where each cell is the mean of the cells it represent
    :param df: The input df
    :param ratio: What square size from the original df each mean will be calculated from.
                  If the side exceeds the df boundary than use a partial square 
    :return The new mean_df
    """
    index, column = df.shape
    log.info("Creating df thumbnail from df ({},{}) with a ratio of {}".format(index, column, ratio))
    means = []
    means_line = []
    for index_i in range(0, index, ratio):
        for column_i in range(0, column, ratio):
            index_start = index_i - ratio//2
            index_start = index_start if index_start > 0 else 0
            column_start = column_i - ratio//2
            column_start = column_start if column_start > 0 else 0
            local_df = df.iloc[index_start:(index_i + ratio//2), column_start:(column_i + ratio//2)]
            means_line.append(local_df.mean().mean())
        means.append(means_line)
        means_line = []
    return pd.DataFrame(means)

def bluntify_each_cell_by_area(df, diffpoints, area_size=None):
    """
    
    :param df: The dataframe to work on
    :param diffpoints: A sorted by "diff" iterable of (diff, color_lower, color_greater) to compare to the mean.
                       Each of the df's cells' value in the range between two diffs from the mean will be changed
                       to the appropriate color (from the lower diffpoint).
                       All values closer than the smallest diffpoint will be changed to np.NAN
    :param area_size: What square size from the original df each mean will be calculated from.
                      If the side exceeds the df boundary than use a partial square.
    """
    msg = "Bluntifying with diffpoints {}".format(diffpoints)
    if area_size:
        msg += " and an area size of {}".format(area_size)
    log.info(msg)
    if area_size is None:
        mean = bluntify(df, diffpoints)
        log.info("Old mean was {}".format(mean))
        return
    ratio = math.ceil(area_size / 3)
    df_means_middle = create_mean_thumbnail(df, ratio)
    df_means = create_mean_thumbnail(df_means_middle, 3)
    log.info("Bluntifying df ({0[0]},{0[1]})".format(df.shape))
    for c, column_name in enumerate(df):
        for i, cell in enumerate(df[column_name]):
            bluntify_cell(df, df_means, i, c, ratio * 3, diffpoints)


def bluntify_cell(df, df_means, index, column, ratio, diffpoints):
    # small_index = index // ratio
    # small_column = column // ratio
    # start_index = small_index - 1 if small_index - 1 > 0 else 0
    # start_column = small_column - 1 if small_column - 1 > 0 else 0
    # local_mean_df = df_means.iloc[start_index:small_index + 2, start_column:small_column + 2]
    # mean = local_mean_df.mean().mean()
    mean = df_means.iloc[index // ratio, column // ratio]
    lower_values, greater_values = generate_diffvalues_from_mean(mean, diffpoints)
    df.iloc[index, column] = find_bluntcolor(df.iloc[index, column], lower_values=lower_values, greater_values=greater_values)

# def bluntify_by_local(df, diffpoints, local_area=None, area_multiplier=None):
#     """
    
#     :param df: The dataframe to work on
#     :param diffpoints: A sorted by "diff" iterable of (diff, color_lower, color_greater) to compare to the mean.
#                        Each of the df's cells' value in the range between two diffs from the mean will be changed
#                        to the appropriate color (from the lower diffpoint).
#                        All values closer than the smallest diffpoint will be changed to np.NAN
#     :param local_area: A (index, column) tuple indicating the size of the local area.
#                        If not given uses the entire df as the area
#     :param area_multiplier: An optional multiplier to increase the area just for mean calculation
#     """
#     msg = "Bluntifying with diffpoints {}".format(diffpoints)
#     if local_area:
#         msg += " and a local area of {}".format(local_area)
#     if area_multiplier:
#         msg += " and a area multiplier of {}".format(area_multiplier)
#     log.info(msg)
#     if local_area:
#         df_cpy = df.copy()
#         index, columns = df.shape
#         for index_i in range(0, index, local_area[0]):
#             for column_i in range(0, columns, local_area[1]):
#                 local_df = df.iloc[index_i:(index_i + local_area[0]), column_i:(column_i + local_area[1])]
#                 mean=None
#                 if area_multiplier is not None:
#                     mean = calc_large_area_mean(df_cpy, local_area, index_i, column_i, area_multiplier)
#                 bluntify(local_df, diffpoints, mean)
#         del df_cpy
#     else:
#         mean = bluntify(df, diffpoints)
#         log.info("Old mean was {}".format(mean))

# def calc_large_area_mean(df, local_area, index, column, area_multiplier):
#     """
#     Calculate the mean in a local_area stating from (index, column) that was mad larger by area_multiplier
#     :param df: The dataframe to work on
#     :param local_area: A (index, column) tuple indicating the size of the local area.
#                        If not given uses the entire df as the area
#     :param index: Start calculation from this index
#     :param column: Start calculation from this column
#     :param area_multiplier: An optional multiplier to increase the area just for mean calculation
#     """
#     if local_area is None:
#         return df.mean().mean()
#     index_diff = int((local_area[0] * area_multiplier) - local_area[0])
#     column_diff = int((local_area[1] * area_multiplier) - local_area[1])
#     large_index_start = index - index_diff
#     large_index_start = large_index_start if large_index_start > 0 else 0
#     large_index_end = index + local_area[0] + index_diff
#     large_column_start = column - column_diff
#     large_column_start = large_column_start if large_column_start > 0 else 0
#     large_column_end = column + local_area[1] + column_diff
#     df_large = df.iloc[large_index_start:large_index_end, large_column_start:large_column_end]
#     return df_large.mean().mean()

def bluntify(df, diffpoints, mean=None):
    """
    
    :param df: The dataframe to work on
    :param diffpoints: A sorted iterable of diffpoints to compare to the mean.
                       Each of the df's cells' value in the range between two diffs from the mean will be changed
                       to the appropriate color (from the lower diffpoint).
                       All values closer than the smallest diffpoint will be changed to np.NAN
    :param mean: An optional custom mean. If not given will generate one instead
    """
    # Generate points by diffing the mean from each diffpoint
    mean = mean if mean is not None else df.mean().mean()
    lower_values, greater_values = generate_diffvalues_from_mean(mean, diffpoints)
    # Apply in place for each cell
    df[:] = df.applymap(partial(find_bluntcolor, lower_values=lower_values, greater_values=greater_values))
    return mean

def generate_diffvalues_from_mean(mean, diffpoints):
    lower_values = []
    greater_values = []
    for d in diffpoints:
        # Keep the lists sorted
        lower_values.insert(0, mean - d)
        greater_values.append(mean + d)
    # Make sure the number of values is not larger than the number of colors
    lower_values = lower_values[:MAX_DIFFPOINTS]
    greater_values = greater_values[:MAX_DIFFPOINTS]
    return lower_values, greater_values

def find_bluntcolor(num, lower_values, greater_values):
    """
    If num is in between the lower and greater values, return np.nan
    Else return the color of the closest "middle" value (by index of color) 
    :param num: The number to compare to
    :param lower_values: A sorted iterable of values lower than the mean
    :param greater_values: A sorted iterable of values greater than the mean
    """
    i_lower = bisect.bisect_right(lower_values, num)
    if i_lower < len(lower_values):
        return COLORS_LOWER[i_lower]
    i_greater = bisect.bisect_left(greater_values, num)
    if i_greater == 0:
        return np.nan
    else:
        return COLORS_GREATER[i_greater - 1]

def df_to_image(df, image_path):
    log.info('Writing image to {}'.format(image_path))
    imsave(image_path, df, cmap=COLOR_MAP, vmin=PAIRED_MIN, vmax=PAIRED_MAX)

def verify_args(args):
    if not args.file_path.is_file():
        raise ValueError('file_path must be a path to file')
    if not args.diffpoint:
        raise ValueError('Use --diffpoint (-d) in order to color (can be used up to {} times)'.format(MAX_DIFFPOINTS))
    if len(args.diffpoint) > MAX_DIFFPOINTS:
        raise ValueError('Surpassed maximum number of diffpoints ({})'.format(MAX_DIFFPOINTS))
    if args.image_path:
        if not args.image_path.parent.is_dir():
            raise ValueError("Invalid image_path! Must be a valid file path")
    if args.image_path and args.image_path.suffix not in VALID_IMAGE_SUFFIXES:
        raise ValueError("Invalid image_path! Suffix must be in {}".format(VALID_IMAGE_SUFFIXES))

def local_area(string):
    try:
         index, column = string.split(',', maxsplit=1)
         return (int(index), int(column))
    except Exception as e:
        raise argparse.ArgumentTypeError(e)

def main(args):
    init_logging(args.log_path)
    df = read_csv(args.file_path, chunksize=None)
    # bluntify_by_local(df, local_area=args.local_area, diffpoints=args.diffpoint, area_multiplier=args.area_multiplier)
    bluntify_each_cell_by_area(df, args.diffpoint, area_size=args.area_size)
    image_path_uniqe = "_local_mean" if args.area_size is not None else "_global_mean"
    image_path = args.image_path if args.image_path else args.file_path.parent / (args.file_path.stem + image_path_uniqe + '.png')
    df_to_image(df, image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turns CSV files into image by local mean')
    parser.add_argument('file_path', type=Path, help='A path to a csv file')
    parser.add_argument('--image-path', '-o', type=Path, help='An optional path to the resulting image. If omitted will generate a path from the input path with a different suffix')
    parser.add_argument('--log-path', '-l', type=Path, help='Save the log to a file instead of stdout')
    parser.add_argument('--diffpoint', '-d', type=float, action='append', help='Add diffpoint to color by.'
                            ' Use multiple times (up to {}) in order to specify multiple colors'.format(MAX_DIFFPOINTS))
    # parser.add_argument('--local-area', '-a', type=local_area, help='An (index, column) to indicate the local size.'
    #                         ' Use 2 numbers separated by a coma')
    parser.add_argument('--area-size', '-s', type=int, help='An int to indicate the local area size.')
    # parser.add_argument('--area-multiplier', '-m', type=float, help='An multiplier to increase the local area just for mean calculation.'
    #                                                                 ' Does nothing if local-area is not given')
    try:
        args = parser.parse_args()
        verify_args(args)
        args.diffpoint.sort()
    except Exception:
        log.exception('Argument validation failed')
        sys.exit(2)
    try:
        main(args)
    except Exception:
        log.exception('An exception was raised:')
        sys.exit(2)
