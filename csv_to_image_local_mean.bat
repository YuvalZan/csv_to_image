@echo off

python "%~dp0\csv_to_image_local_mean.py" -d 50000 -d 100000 -d 150000 -s 400,400 %1
