# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools

# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests

# computation libraries
import numpy as np
import pandas

# coordinate systems
import pyproj 

# statistics
import statsmodels.api as sm

# plotting
import bokeh.charts
import bokeh.io
import bokeh.plotting
import bokeh.tile_providers
import bokeh.palettes

# displaying things
from ipywidgets import Image
import IPython.display

# Some coordinate systems
WEBMERCATOR = pyproj.Proj(init='epsg:3857')
WGS84 = pyproj.Proj(init='epsg:4326')

# If this notebook is not showing up with figures, you can use the following url:
# https://nbviewer.ipython.org/github/openearth/notebooks/blob/master/sealevelmonitor.ipynb
bokeh.io.output_notebook()

urls = {
    'metric_monthly': 'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
    'rlr_monthly': 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_monthly.zip',
    'rlr_annual': 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'
}
dataset_name = 'rlr_annual'

# these compute the rlr back to NAP (ignoring the undoing of the NAP correction)
main_stations = {
    20: {
        'name': 'Vlissingen', 
        'rlr2nap': lambda x: x - (6976-46)
    },
    22: {
        'name': 'Hoek van Holland', 
        'rlr2nap': lambda x:x - (6994 - 121)
    },
    23: {
        'name': 'Den Helder', 
        'rlr2nap': lambda x: x - (6988-42)
    },
    24: {
        'name': 'Delfzijl', 
        'rlr2nap': lambda x: x - (6978-155)
    },
    25: {
        'name': 'Harlingen', 
        'rlr2nap': lambda x: x - (7036-122)
    },
    32: {
        'name': 'IJmuiden', 
        'rlr2nap': lambda x: x - (7033-83)
    }
}

# the main stations are defined by their ids
main_stations_idx = list(main_stations.keys())
main_stations_idx

# download the zipfile
resp = requests.get(urls[dataset_name])

# we can read the zipfile
stream = io.BytesIO(resp.content)
zf = zipfile.ZipFile(stream)

# this list contains a table of 
# station ID, latitude, longitude, station name, coastline code, station code, and quality flag
csvtext = zf.read('{}/filelist.txt'.format(dataset_name))

stations = pandas.read_csv(
    io.BytesIO(csvtext), 
    sep=';',
    names=('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'),
    converters={
        'name': str.strip,
        'quality': str.strip
    }
)
stations = stations.set_index('id')

# the dutch stations in the PSMSL database, make a copy
# or use stations.coastline_code == 150 for all dutch stations
selected_stations = stations.ix[main_stations_idx].copy()
# set the main stations, this should be a list of 6 stations
selected_stations



# show all the stations on a map

# compute the bounds of the plot
sw = (50, -5)
ne = (55, 10)
# transform to web mercator
sw_wm = pyproj.transform(WGS84, WEBMERCATOR, sw[1], sw[0])
ne_wm = pyproj.transform(WGS84, WEBMERCATOR, ne[1], ne[0])
# create a plot
fig = bokeh.plotting.figure(tools='pan, wheel_zoom', plot_width=600, plot_height=200, x_range=(sw_wm[0], ne_wm[0]), y_range=(sw_wm[1], ne_wm[1]))
fig.axis.visible = False
# add some background tiles
fig.add_tile(bokeh.tile_providers.STAMEN_TERRAIN)
# add the stations
x, y = pyproj.transform(WGS84, WEBMERCATOR, np.array(stations.lon), np.array(stations.lat))
fig.circle(x, y)
x, y = pyproj.transform(WGS84, WEBMERCATOR, np.array(selected_stations.lon), np.array(selected_stations.lat))
_ = fig.circle(x, y, color='red')

# show the plot
bokeh.io.show(fig)

# each station has a number of files that you can look at.
# here we define a template for each filename

# stations that we are using for our computation
# define the name formats for the relevant files
names = {
    'datum': '{dataset}/RLR_info/{id}.txt',
    'diagram': '{dataset}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.rlrdata',
    'doc': '{dataset}/docu/{id}.txt',
    'contact': '{dataset}/docu/{id}_auth.txt'
}

def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    url = names['url'].format(**info)
    return url
# fill in the dataset parameter using the global dataset_name
f = functools.partial(get_url, dataset=dataset_name)
# compute the url for each station
selected_stations['url'] = selected_stations.apply(f, axis=1)
selected_stations

def missing2nan(value, missing=-99999):
    """convert the value to nan if the float of value equals the missing value"""
    value = float(value)
    if value == missing:
        return np.nan
    return value

def get_data(station, dataset):
    """get data for the station (pandas record) from the dataset (url)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    bytes = zf.read(names['data'].format(**info))
    df = pandas.read_csv(
        io.BytesIO(bytes), 
        sep=';', 
        names=('year', 'height', 'interpolated', 'flags'),
        converters={
            "height": lambda x: main_stations[station.name]['rlr2nap'](missing2nan(x)),
            "interpolated": str.strip,
        }
    )
    df['station'] = station.name
    return df

# get data for all stations
f = functools.partial(get_data, dataset=dataset_name)
# look up the data for each station
selected_stations['data'] = [f(station) for _, station in selected_stations.iterrows()]

# we now have data for each station
selected_stations[['name', 'data']]

# compute the mean
grouped = pandas.concat(selected_stations['data'].tolist())[['year', 'height']].groupby('year')
mean_df = grouped.mean().reset_index()
# filter out non-trusted part (before NAP)
mean_df = mean_df[mean_df['year'] >= 1890].copy()

# these are the mean waterlevels 
mean_df.tail()

# show all the stations, including the mean
title = 'Sea-surface height for Dutch tide gauges [{year_min} - {year_max}]'.format(
    year_min=mean_df.year.min(),
    year_max=mean_df.year.max() 
)
fig = bokeh.plotting.figure(title=title, x_range=(1860, 2020), plot_width=900, plot_height=400)
colors = bokeh.palettes.Accent6
for color, (id_, station) in zip(colors, selected_stations.iterrows()):
    data = station['data']
    fig.circle(data.year, data.height, color=color, legend=station['name'], alpha=0.5)
fig.line(mean_df.year, mean_df.height, line_width=3, alpha=0.7, color='black', legend='Mean')
fig.legend.location = "bottom_right"
fig.yaxis.axis_label = 'waterlevel [mm] above NAP'
fig.xaxis.axis_label = 'year'


bokeh.io.show(fig)



# define the statistical model
y = mean_df['height']
X = np.c_[
    mean_df['year']-1970, 
    np.cos(2*np.pi*(mean_df['year']-1970)/18.613),
    np.sin(2*np.pi*(mean_df['year']-1970)/18.613)
]
X = sm.add_constant(X)
model = sm.OLS(y, X)
fit = model.fit()

fit.summary(yname='Sea-surface height', xname=['Constant', 'Trend', 'Nodal U', 'Nodal V'])
# things to check:
# Durbin Watson should be >1 for no worries, >2 for no autocorrelation
# JB should be non-significant for normal residuals
# abs(x2.t) + abs(x3.t) should be > 3, otherwise adding nodal is not useful

fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
for color, (id_, station) in zip(colors, selected_stations.iterrows()):
    data = station['data']
    fig.circle(data.year, data.height, color=color, legend=station['name'], alpha=0.8)
fig.circle(mean_df.year, mean_df.height, line_width=3, legend='Mean', color='black', alpha=0.5)
fig.line(mean_df.year, fit.predict(), line_width=3, legend='Current')
fig.legend.location = "bottom_right"
fig.yaxis.axis_label = 'waterlevel [mm] above N.A.P.'
fig.xaxis.axis_label = 'year'
bokeh.io.show(fig)

# define the statistical model
y = mean_df['height']
X = np.c_[
    mean_df['year']-1970, 
    (mean_df['year'] > 1993) * (mean_df['year'] - 1993),
    np.cos(2*np.pi*(mean_df['year']-1970)/18.613),
    np.sin(2*np.pi*(mean_df['year']-1970)/18.613)
]
X = sm.add_constant(X)
model_broken_linear = sm.OLS(y, X)
fit_broken_linear = model_broken_linear.fit()

# define the statistical model
y = mean_df['height']
X = np.c_[
    mean_df['year']-1970, 
    (mean_df['year'] - 1970) * (mean_df['year'] - 1970),
    np.cos(2*np.pi*(mean_df['year']-1970)/18.613),
    np.sin(2*np.pi*(mean_df['year']-1970)/18.613)
]
X = sm.add_constant(X)
model_quadratic = sm.OLS(y, X)
fit_quadratic = model_quadratic.fit()

fit_broken_linear.summary(yname='Sea-surface height', xname=['Constant', 'Trend', 'Trend(year > 1990)', 'Nodal U', 'Nodal V'])

fit_quadratic.summary(yname='Sea-surface height', xname=['Constant', 'Trend', 'Trend**2', 'Nodal U', 'Nodal V'])

fig = bokeh.plotting.figure(x_range=(1860, 2020), plot_width=900, plot_height=400)
for color, (id_, station) in zip(colors, selected_stations.iterrows()):
    data = station['data']
    fig.circle(data.year, data.height, color=color, legend=station['name'], alpha=0.8)
fig.circle(mean_df.year, mean_df.height, line_width=3, legend='Mean', color='black', alpha=0.5)
fig.line(mean_df.year, fit.predict(), line_width=3, legend='Current')
fig.line(mean_df.year, fit_broken_linear.predict(), line_width=3, color='#33bb33', legend='Broken')
fig.line(mean_df.year, fit_quadratic.predict(), line_width=3, color='#3333bb', legend='Quadratic')

fig.legend.location = "top_left"
fig.yaxis.axis_label = 'waterlevel [mm] above N.A.P.'
fig.xaxis.axis_label = 'year'
bokeh.io.show(fig)

msg = '''The current average waterlevel above NAP (in mm), 
based on the 6 main tide gauges for the year {year} is {height:.1f} cm.
The current sea-level rise is {rate:.0f} cm/century'''
print(msg.format(year=mean_df['year'].iloc[-1], height=fit.predict()[-1]/10.0, rate=fit.params.x1*100.0/10))

if (fit.aic < fit_broken_linear.aic):
    print('The linear model is a higher quality model (smaller AIC) than the broken linear model.')
else:
    print('The broken linear model is a higher quality model (smaller AIC) than the linear model.')
if (fit_broken_linear.pvalues['x2'] < 0.05):
    print('The trend break is bigger than we would have expected under the assumption that there was no trend break.')
else:
    print('Under the assumption that there is no trend break, we would have expected a trend break as big as we have seen.')

if (fit.aic < fit_quadratic.aic):
    print('The linear model is a higher quality model (smaller AIC) than the quadratic model.')
else:
    print('The quadratic model is a higher quality model (smaller AIC) than the linear model.')
if (fit_quadratic.pvalues['x2'] < 0.05):
    print('The quadratic term is bigger than we would have expected under the assumption that there was no quadraticness.')
else:
    print('Under the assumption that there is no quadraticness, we would have expected a quadratic term as big as we have seen.')



