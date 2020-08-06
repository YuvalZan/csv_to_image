@echo off

echo generating double_rainbow
python generate_double_rainbow.py
echo generating generate_sheet
python generate_sheet.py
echo generating generate_scratched_sheet
python generate_scratched_sheet.py

echo imaging double_rainbow
python ..\csv_to_image.py double_rainbow.csv
echo imaging sheet
python ..\csv_to_image.py -m 10 -x 40 sheet.csv
echo imaging scratched_sheet
python ..\csv_to_image.py -m 10 -x 40 scratched_sheet.csv
echo Done