from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.pylab import imsave

# TABLE_PATH = Path(r'C:\Users\zanyu\Documents\git\csv_to_image\test_data.csv')
TABLE_PATH = Path(r'~\Documents\git\csv_to_image\test_data\test_data.csv')
TABLE_PATH = Path(r'~\Documents\git\csv_to_image\test_data/zap1000.csv')
TABLE_PATH = Path(r'~\Documents\git\csv_to_image\test_data/iphone data.csv')
TABLE_PATH = TABLE_PATH.expanduser()
TABLE_PATH2 = TABLE_PATH.parent / (TABLE_PATH.stem + "_2" + TABLE_PATH.suffix)
TABLE_IMAGE = TABLE_PATH.parent / (TABLE_PATH.stem + '.png')



df = table = pd.read_csv(TABLE_PATH, header=None, dtype=np.float32, float_precision='high')
# df = table = pd.read_csv(TABLE_PATH, header=None, dtype=np.float32, float_precision='high', chunksize=100)
df = pd.read_csv('tests/sheet.csv', dtype=np.float16, index_col=0, engine='python')


df[df < 0] = 0


smallest = table.min().min()
biggest = table.max().max()
print('Biggest diff is', biggest - smallest)
table -= smallest

table = (table - table.mean()) / table.std()


df[df > 25] = np.nan
df[df.isnull()] = 0


imsave(TABLE_IMAGE, table)


import matplotlib.image as mpimg
im = mpimg.imread(r'C:\Users\zanyu\Documents\git\csv_to_image\test_data\zap1000_chunk100_00.png')



def merge_images(*images_path):
	import sys
	from PIL import Image

	images = [Image.open(x) for x in images_path]
	widths, heights = zip(*(i.size for i in images))

	max_width = max(widths)
	total_height = sum(heights)

	new_im = Image.new('RGB', (max_width, total_height))
	x_offset = 0
	for im in images:
	  new_im.paste(im, (0, x_offset))
	  x_offset += im.size[1]

	new_im.save('merged.png')



### Create test data
from itertools import chain, repeat, accumulate
line = chain(range(2500), range(2500, 0, -1))
line = [(cell/500) + 20 for cell in line]
# line_generator = chain(range(2500), range(2500, 0, -1))
table = repeat(line, 5000)
test = pd.DataFrame(table, dtype=np.float16)
# test.to_csv(r'%userprofile%\Documents\git\csv_to_image\test_data.csv')
test.to_csv(TABLE_PATH, header=False, index=False)
# test.to_excel(r'C:\Users\zanyu\Documents\git\csv_to_image\test_data.xlsx', header=False)

# table.to_csv(TABLE_PATH2, header=False, index=False)
###/



nuitka --plugin-list

set MATPLOTLIBDATA="C:\program files (x86)\python37-32\lib\site-packages\matplotlib\mpl-data"

# nuitka  --follow-imports --standalone --plugin-enable=numpy --plugin-enable=tk-inter csv_to_image.py 
nuitka  --follow-imports --standalone --plugin-enable=numpy --plugin-enable=tk-inter csv_to_image.py --show-progress --show-modules --windows-icon=vision.ico --include-package=matplotlib > log_out.txt 2> log_err.txt

--python-arch="x86_64"
--follow-stdlib
--windows-disable-console


NUITKA_CLCACHE_BINARY="C:\Program Files (x86)\Python37-32\Scripts\clcache.exe"
nuitka-hints.py --standalone --windows-icon=vision.ico --show-progress csv_to_image.py

nuitka-hints.py --standalone --windows-icon=vision.ico csv_to_image.py
--windows-dependency-tool=pefile
--experimental=use_pefile_recurse
--experimental=use_pefile_fullrecurse

--show-progress --show-modules







