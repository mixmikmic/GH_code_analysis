get_ipython().magic('matplotlib inline')
from __future__ import print_function
import numpy as np

import Utilities.nctools as nctools
import Utilities.metutils as metutils
from Utilities.maputils import find_index

from PlotInterface.maps import HazardMap, FilledContourMapFigure, saveFigure
from matplotlib.backends.backend_nbagg import FigureCanvasNbAgg as FigureCanvas

import seaborn as sns
sns.set_context('talk')

class PlotUnits(object):

    def __init__(self, units):
        labels = {
            'mps': 'm/s',
            'm/s': 'm/s',
            'mph': 'mi/h',
            'kts': 'kts',
            'kph': 'km/h',
            'kmh': 'km/h'
        }

        levels = {
            'mps': np.arange(30, 101., 5.),
            'm/s': np.arange(30, 101., 5.),
            'mph': np.arange(80, 221., 10.),
            'kts': np.arange(60, 201., 10.),
            'kph': np.arange(80, 361., 20.),
            'kmh': np.arange(80, 361., 20.)
        }

        self.units = units
        self.label = labels[units]
        self.levels = levels[units]

unit_opts = ['m/s', 'mph', 'kts', 'kmh']

url = "http://dapds00.nci.org.au/thredds/dodsC/fj6/TCRM/benchmark/hazard/hazard.nc"
ncobj = nctools.ncLoadFile(url)
lon = nctools.ncGetDims(ncobj, 'lon')[:]
lat = nctools.ncGetDims(ncobj, 'lat')[:]
yrs = nctools.ncGetDims(ncobj, 'return_period')[:]
years = [str(int(y)) for y in yrs]
wspd = nctools.ncGetVar(ncobj, 'wspd')
wspdUnits = wspd.units

try:
    wLower  = nctools.ncGetVar(ncobj, 'wspdlower')
    wUpper = nctools.ncGetVar(ncobj, 'wspdupper')
    ciBounds = True
except KeyError:
    ciBounds = False

minLon = min(lon)
maxLon = max(lon)
minLat = min(lat)
maxLat = max(lat)
xlon, ylat = np.meshgrid(lon, lat)

map_kwargs = dict(llcrnrlon=minLon, llcrnrlat=minLat, 
                  urcrnrlon=maxLon, urcrnrlat=maxLat,
                  projection='merc', resolution='i')

def plot_map(year, units):
    pU = PlotUnits(units)
    idx = years.index(year)
    data = wspd[idx, :, :]
    fig = HazardMap()
    title = "{0}-year ARI wind speed".format(year)
    cbarlabel = "Wind speed ({0})".format(pU.label)
    levels = pU.levels
    
    fig.plot(metutils.convert(data, wspd.units, pU.units), 
             xlon, ylat, title, levels, cbarlabel, map_kwargs)
    return fig

fig = plot_map('500', 'mps')
fig.set_size_inches(16, 8)
canvas = FigureCanvas(fig)
fig

