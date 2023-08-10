import pandas as pd

df = pd.read_csv("../data/pollution_us_2000_2016.csv")

from os import path
import sys
sys.path.append(path.abspath('../tests'))

source = 'CO' # options: NO2, O3, SO2 and CO
year = '2008' # options: 2000 - 2016
option = 'Mean' # options: Mean, AQI, 1st Max Value

from pollution_map import pollution_map 
fig1 = pollution_map(df, source, year, option)

from pollution_change import pollution_change
fig2 = pollution_change(df, source, year, option)



