from itertools import chain, repeat
import datetime
import numpy as np
import pandas as pd

DTYPE = np.float16
ROWS = 2500
COLUMNS = 2500
BASE_VALUE = 20 
MAX_DIFF = 5

def generate_double_rainbow(rows, columns):
	line = chain(range(int(rows/2)), range(int(rows/2), 0, -1))
	line = [(cell/(rows/MAX_DIFF)) + BASE_VALUE for cell in line]
	table = repeat(line, columns)
	return pd.DataFrame(table, dtype=DTYPE)

def main():
	print(datetime.datetime.now(), "Started generate_double_rainbow")
	generate_double_rainbow(ROWS, COLUMNS).to_csv('double_rainbow.csv')
	print(datetime.datetime.now(), "Done generate_double_rainbow")


if __name__ == '__main__':
	main()