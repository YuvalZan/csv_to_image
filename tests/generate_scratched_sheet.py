import datetime
from generate_sheet import generate_sheet, ROWS, COLUMNS

SCRATCH_LENGTH = 200
SCRATCH_THIKNESS = 3
SCRATCH_MODIFIER = 10

def scratch_sheet(df, start, length, thikness, modifier, direction=(1, 1)):
    for j in range(thikness):
        for i in range(length):
            x = start + (i + j)*direction[0]
            y = start + i * direction[1]
            df[x][y] += modifier
    return df



def main():
    print(datetime.datetime.now(), "Started generate_scratched_sheet")
    df = generate_sheet(ROWS, COLUMNS)
    scratch_sheet(df, COLUMNS / 4, SCRATCH_LENGTH, SCRATCH_THIKNESS, SCRATCH_MODIFIER)
    scratch_sheet(df, COLUMNS*3 / 4 + SCRATCH_LENGTH, SCRATCH_LENGTH, SCRATCH_THIKNESS, -SCRATCH_MODIFIER, direction=(-1, -1))
    df.to_csv('scratched_sheet.csv')
    print(datetime.datetime.now(), "Done generate_scratched_sheet")


if __name__ == '__main__':
    main()