import os
import numpy as np
import pandas as pd
import gdal

get_ipython().magic('run plot_and_table_functions.py')

# This cell actually runs the SWB2 model for the Central Sands;
# execution time will probably be several minutes. Wait until the asterisk ('*') 
# changes to a number; normal output 'Out[#]' is '0'.
dirpath = os.getcwd().split('/')
if dirpath[-1] == 'ipython':
    os.chdir('..')
os.system('swb2 --output_prefix=central_sands_ --output_dir=output central_sands_swb2.ctl')

# need to do the NetCDF thing here.....

