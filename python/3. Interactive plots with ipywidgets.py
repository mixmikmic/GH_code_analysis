import os
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
get_ipython().magic('matplotlib inline')

# import ipywidgets
from ipywidgets import *

mpl.rcParams['figure.facecolor'] = 'white'

# pandas options
pd.options.display.max_rows = 6

all_depths = pd.read_pickle(r'..\data\Groundwater-Levels\all_depths_v1.pkl.gz')

all_depths1 = all_depths.loc[(slice(None),1),:].reset_index(1, drop=True)
all_depths1_permonth = all_depths1.reset_index(0).groupby('well').resample('MS').mean()

# first define the dimensions of the grid for the depth surface
x_min = all_depths.x.min()
x_max = all_depths.x.max()
y_min = all_depths.y.min()
y_max = all_depths.y.max()

cellsize = 100
vector_x = np.arange(x_min-2*cellsize, x_max+3*cellsize, cellsize)
vector_y = np.arange(y_min-2*cellsize, y_max+3*cellsize, cellsize)
grid_x, grid_y = np.meshgrid(vector_x,vector_y)
extent1 = (vector_x.min(),vector_x.max(), vector_y.min(), vector_y.max())

def plot_surface(date='1996-01-01', method='nearest', add_wells=True):
    # selection and gridding
    dfa = all_depths1_permonth.loc[(slice(None),date),:].dropna()
    points = dfa[['x','y']]
    values = dfa.depth
    grid1 = interpolate.griddata(points, values, (grid_x, np.flipud(grid_y)), method=method)
    # plotting
    fig,ax = plt.subplots(figsize=(16,12))
    plt.imshow(grid1,extent=extent1,origin='upper', cmap='coolwarm_r',
               vmin=all_depths1.depth.quantile(0.04) ,
               vmax=all_depths1.depth.quantile(0.96))  # fix the depth range to fix the colormap
    if add_wells: 
        dfa.plot.scatter(x='x', y='y', c='none', s=50,edgecolors='k', ax=ax, colorbar=False)
    # 
    plt.colorbar(shrink=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(date)
    plt.show() # this is necessary to ensure the same plot is updated as we change the date with the cursor

# create a list of dates (every first day of the month from 1955 to 2015)
dates = [datetime.date(y,m,1) for y in range(1955,2016) for m in range(1,13) ]
# create a widget with a slider for the data
dateWidget = SelectionSlider(
                options=dates,
                description='Date')
# make the slider wider
dateWidget.layout.width = '500px'
# make a couple of radio buttons for the method selection
methodWidget = RadioButtons(
            options=['nearest', 'linear'], value='nearest',
            description='Interpolation:')
# run widget with the interact function (note how the function automatically creates a tick box for the add_wells parameter)
interact(plot_surface, date=dateWidget, method=methodWidget)

def plot_depth_vs_time(well='B24F0005'):
    # selection 
    dfa = all_depths1_permonth.loc[well]
    
    # plot
    fig,ax = plt.subplots(figsize=(12,6))
    ax.plot(dfa.depth)
    ax.set_xlabel('')
    ax.set_ylabel('mean depth (cm w.r.t. NAP)')
    ax.set_xlim(pd.to_datetime('1950'), pd.to_datetime('2017'))
    ax.set_ylim(-750,750)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.grid()
    plt.show()

# create a list of all the wells
well_list = all_depths1.index.get_level_values(0).unique().tolist()
# create a widget with a dropdown menu
wellWidget = Dropdown(options=well_list, value=well_list[0],
                description='Well ID')

# run widget with the interact function
interact(plot_depth_vs_time, well=wellWidget)

