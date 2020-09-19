import datetime
from generate_sheet import generate_sheet, ROWS, COLUMNS
from generate_scratched_sheet import scratch_sheet, SCRATCH_LENGTH, SCRATCH_MODIFIER, SCRATCH_THIKNESS

DISTORT_INDEX_MODIFIER = 20
DISTORT_COLUMN_MODIFIER = 8

def distort_sheet(df, index_modifier, column_modifier):
    num_indexes, num_columns = df.shape
    column_step = column_modifier / num_columns
    for i in range(0, num_columns):
        df.iloc[:, i] += column_step * i
    index_step = index_modifier / num_indexes
    for i in range(0, num_indexes):
        df.iloc[i] += index_step * i

    return df

def main():
    print(datetime.datetime.now(), "Started generate_distorted_sheet")
    df = generate_sheet(ROWS, COLUMNS)
    scratch_sheet(df, COLUMNS / 4, SCRATCH_LENGTH, SCRATCH_THIKNESS, SCRATCH_MODIFIER)
    scratch_sheet(df, COLUMNS*3 / 4 + SCRATCH_LENGTH, SCRATCH_LENGTH, SCRATCH_THIKNESS, -SCRATCH_MODIFIER, direction=(-1, -1))
    distort_sheet(df, DISTORT_INDEX_MODIFIER, DISTORT_COLUMN_MODIFIER)
    df.to_csv('distorted_sheet.csv')
    print(datetime.datetime.now(), "Done generate_distorted_sheet")


if __name__ == '__main__':
    main()