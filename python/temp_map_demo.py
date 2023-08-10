import zipfile
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from os import path
import sys
sys.path.append(path.abspath('../tests'))

zf = zipfile.ZipFile('../data/GlobalLandTemperatures.zip')
temp = pd.read_csv(zf.open('GlobalLandTemperaturesByState.csv'))

from temp_map import temp_map

# Specify a year (1900-2013)
year = '1980'

fig1 = temp_map(temp, year)

from temp_increase import temp_increase

# From which year you'd like to compare (1900-2012)
year = '1980'

fig2 = temp_increase(temp, year)



