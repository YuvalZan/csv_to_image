import datetime
import numpy as np
import pandas as pd

DTYPE = np.float16
ROWS = 2500
COLUMNS = 2500
BASE_VALUE = 20
MAX_DIFF = 0.3

def generate_sheet(rows, columns):
	df = pd.DataFrame(np.random.random_sample((rows, columns)))
	df *= MAX_DIFF
	df += BASE_VALUE
	return df

def main():
	print(datetime.datetime.now(), "Started generate_sheet")
	generate_sheet(ROWS, COLUMNS).to_csv('sheet.csv')
	print(datetime.datetime.now(), "Done generate_sheet")


if __name__ == '__main__':
	main()