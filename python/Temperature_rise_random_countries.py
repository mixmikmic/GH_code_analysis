import wget
import os
import zipfile

import urllib3
import certifi
import sys
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns

import plotly
plotly.offline.init_notebook_mode()
  

zf = zipfile.ZipFile('../data/GlobalLandTemperatures.zip')

countries = pd.read_csv(zf.open('GlobalLandTemperaturesByCountry.csv'), parse_dates=['dt'])
countries.Country = countries.Country.str.cat(countries.Country, sep=' ')
countries = countries[countries.dt.dt.year >= 1900]
countries.head()

splits = countries['Country'].str.split()
countries['Countries'] = splits.str[0]
countries.head()

country_means = countries.groupby(['Countries', countries.dt.dt.year])['AverageTemperature'].mean().unstack()
country_mins = countries.groupby(['Countries', countries.dt.dt.year])['AverageTemperature'].min().unstack()
country_maxs = countries.groupby(['Countries', countries.dt.dt.year])['AverageTemperature'].max().unstack()
country_means.head()

first_years_mean = country_means.iloc[:, :5].mean(axis=1) # mean temperature for the first 5 years
country_means_shifted = country_means.subtract(first_years_mean, axis=0)

def plot_temps(countries, country_ser, ax):
    first_years_mean = country_ser.iloc[:, :5].mean(axis=1)
    country_ser = country_ser.subtract(first_years_mean, axis=0)
    for city in random_countries:
        row = country_ser.loc[city]
        pd.stats.moments.ewma(row, 10).plot(label=row.name, ax=ax)
    ax.set_xlabel('')
    ax.legend(loc='best')

fig, axes = plt.subplots(3,1, figsize=(10,10))

n = 5 # number of random countries you want to see
random_countries = country_means_shifted.sample(n).index

plot_temps(random_countries, country_means, axes[0])
plot_temps(random_countries, country_mins, axes[1])
plot_temps(random_countries, country_maxs, axes[2])

axes[0].set_title("Year's mean temperature increase for random countries")
axes[1].set_title("Year's min temperature increase for random countries")
axes[2].set_title("Year's max temperature increase  for random countries")



