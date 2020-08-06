import datetime
from generate_sheet import generate_sheet, ROWS, COLUMNS

SCRATCH_LENGTH = 20
SCRATCH_MODIFIER = 10

def generate_scratched_sheet(rows, columns):
	df = generate_sheet(ROWS, COLUMNS)
	start = COLUMNS / 4
	for i in range(SCRATCH_LENGTH):
		df[start + i][start + i] += SCRATCH_MODIFIER

	return df

def main():
	print(datetime.datetime.now(), "Started generate_scratched_sheet")
	generate_scratched_sheet(ROWS, COLUMNS).to_csv('scratched_sheet.csv')
	print(datetime.datetime.now(), "Done generate_scratched_sheet")


if __name__ == '__main__':
	main()