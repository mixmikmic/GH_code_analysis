import pandas as pd

df = pd.read_csv("../data/pollution_us_2000_2016.csv")

from os import path
import sys
sys.path.append(path.abspath('../tests'))

year="2000"
state="Arizona"

from plot_pollutants import plot_pollutants
fig= plot_pollutants(df, year, state)



