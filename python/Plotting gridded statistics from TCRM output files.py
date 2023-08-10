get_ipython().magic('matplotlib inline')
from Utilities.nctools import ncLoadFile, ncGetDims, ncGetVar, ncFileInfo
from Utilities.grid import SampleGrid
from PlotInterface.maps import ArrayMapFigure, saveFigure, levels
import numpy as np
from IPython.display import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FC

from os.path import join as pjoin
import seaborn as sns
sns.set_context('paper')

def show(fig):
    canvas = FC(fig)
    fig

landmask_file = "C:/WorkSpace/tcrm/input/landmask.nc"
landmask = SampleGrid(landmask_file)

def getData(ncobj, varname, ij):
    var = ncGetVar(ncobj, varname)[:]
    lvar = ncGetVar(ncobj, 'l'+varname)[:]
    data = var
    data[ij] = lvar[ij]
    return data

path = "C:/Workspace/temp"
fname = pjoin(path, "pressure_rate_stats.nc")
ncFileInfo(fname)

ncobj = ncLoadFile(fname)

#lon = np.arange(90., 180., 1.0)
#lat = np.arange(-30., -5., 1.0)

lon = ncGetDims(ncobj, 'lon')
lat = ncGetDims(ncobj, 'lat')
ncobj.close()
xgrid, ygrid = np.meshgrid(lon, lat)

ls = np.zeros(np.shape(xgrid))

for i in range(len(lon)):
    for j in range(len(lat)):
        if landmask.sampleGrid(lon[i], lat[j]) > 0.0:
            ls[j, i] = 1
            
ij = np.where(ls==1)

map_kwargs = dict(llcrnrlon=100., llcrnrlat=-30,
                  urcrnrlon=160., urcrnrlat=-5.,
                  resolution='h', projection='merc')

ncobj = ncLoadFile(pjoin(path, "pressure_rate_stats.nc"))
lon = ncGetDims(ncobj, 'lon')
lat = ncGetDims(ncobj, 'lat')
ardata = getData(ncobj, 'alpha', ij)
mudata = getData(ncobj, 'mu', ij)
mindata = getData(ncobj, 'min', ij)
sigdata = getData(ncobj, 'sig', ij)
ncobj.close()
fig = ArrayMapFigure()
fig.add(ardata, xgrid, ygrid, 'Pressure rate AR(1)', [-1, 1], 'AR(1)', map_kwargs)
fig.add(mudata, xgrid, ygrid, 'Mean pressure rate', [-1, 1], 'Pressure rate (hPa/hr)', map_kwargs)
fig.add(mindata, xgrid, ygrid, 'Minimum pressure rate', [-10, 10], 'Pressure rate (hPa/hr)', map_kwargs)
fig.add(sigdata, xgrid, ygrid, 'Pressure rate standard deviation', [0, 5], 'Std dev.', map_kwargs)
fig.plot()
#canvas = FC(fig)
#fig
saveFigure(fig, "pressure_rate_stats.png")
Image("pressure_rate_stats.png")

ncobj = ncLoadFile(pjoin(path, "pressure_stats.nc"))
lon = ncGetDims(ncobj, 'lon')
lat = ncGetDims(ncobj, 'lat')
ardata = getData(ncobj, 'alpha', ij)
mudata = getData(ncobj, 'mu', ij)
mindata = getData(ncobj, 'min', ij)
sigdata = getData(ncobj, 'sig', ij)
ncobj.close()
fig = ArrayMapFigure()
fig.add(ardata, xgrid, ygrid, 'Pressure AR(1)', [-1, 1], 'AR(1)', map_kwargs)
fig.add(mudata, xgrid, ygrid, 'Mean pressure ', [950, 1000], 'Pressure (hPa)', map_kwargs)
fig.add(mindata, xgrid, ygrid, 'Minimum pressure', [900, 1000], 'Pressure (hPa)', map_kwargs)
fig.add(sigdata, xgrid, ygrid, 'Pressure standard deviation', [0, 50], 'Std dev.', map_kwargs)
fig.plot()
#canvas = FC(fig)
#fig
saveFigure(fig, "pressure_stats.png")
Image("pressure_stats.png")

ncobj = ncLoadFile(pjoin(path, "speed_rate_stats.nc"))
lon = ncGetDims(ncobj, 'lon')
lat = ncGetDims(ncobj, 'lat')
ardata = getData(ncobj, 'alpha', ij)
mudata = getData(ncobj, 'mu', ij)
mindata = getData(ncobj, 'min', ij)
sigdata = getData(ncobj, 'sig', ij)
ncobj.close()
fig = ArrayMapFigure()
fig.add(ardata, xgrid, ygrid, 'Speed rate AR(1)', [-1, 1], 'AR(1)', map_kwargs)
fig.add(mudata, xgrid, ygrid, 'Mean speed rate', [-1, 1], 'Speed rate (m/s/hr)', map_kwargs)
fig.add(mindata, xgrid, ygrid, 'Minimum speed rate', [-10, 10], 'Speed rate (m/s/hr)', map_kwargs)
fig.add(sigdata, xgrid, ygrid, 'Speed standard deviation', [0, 5], 'Std dev.', map_kwargs)
fig.plot()
#canvas = FC(fig)
#fig
saveFigure(fig, "speed_rate_stats.png")
Image("speed_rate_stats.png")

ncobj = ncLoadFile(pjoin(path, "speed_stats.nc"))
lon = ncGetDims(ncobj, 'lon')
lat = ncGetDims(ncobj, 'lat')
ardata = getData(ncobj, 'alpha', ij)
mudata = getData(ncobj, 'mu', ij)
mindata = getData(ncobj, 'min', ij)
sigdata = getData(ncobj, 'sig', ij)
ncobj.close()
fig = ArrayMapFigure()
fig.add(ardata, xgrid, ygrid, 'Speed AR(1)', [-1, 1], 'AR(1)', map_kwargs)
fig.add(mudata, xgrid, ygrid, 'Mean speed', [0, 25], 'Speed (m/s)', map_kwargs)
fig.add(mindata, xgrid, ygrid, 'Minimum speed', [0, 25], 'Speed (m/s)', map_kwargs)
fig.add(sigdata, xgrid, ygrid, 'Speed standard deviation', [0, 20], 'Std dev.', map_kwargs)
fig.plot()
#canvas = FC(fig)
#fig
saveFigure(fig, "speed_stats.png")
Image("speed_stats.png")

ncobj = ncLoadFile(pjoin(path, "bearing_rate_stats.nc"))
lon = ncGetDims(ncobj, 'lon')
lat = ncGetDims(ncobj, 'lat')
ardata = getData(ncobj, 'alpha', ij)
mudata = getData(ncobj, 'mu', ij)
mindata = getData(ncobj, 'min', ij)
sigdata = getData(ncobj, 'sig', ij)
ncobj.close()

print np.percentile(sigdata, 90)
print np.median(sigdata)

fig = ArrayMapFigure()
fig.add(ardata, xgrid, ygrid, 'Bearing rate AR(1)', [-1, 1], 'AR(1)', map_kwargs)
fig.add(mudata, xgrid, ygrid, 'Mean bearing rate', [0, 5], 'Bearing rate (degrees/hr)', map_kwargs)
fig.add(mindata, xgrid, ygrid, 'Minimum bearing rate', [-1., 1], 'Bearing rate (degrees/hr)', map_kwargs)
fig.add(sigdata, xgrid, ygrid, 'Bearing rate standard deviation', [0, 15], 'Std dev.', map_kwargs)
fig.plot()
#canvas = FC(fig)
#fig
saveFigure(fig, "bearing_rate_stats.png")
Image("bearing_rate_stats.png")

ncobj = ncLoadFile(pjoin(path, "bearing_stats.nc"))
lon = ncGetDims(ncobj, 'lon')
lat = ncGetDims(ncobj, 'lat')
ardata = getData(ncobj, 'alpha', ij)
mudata = getData(ncobj, 'mu', ij)
mindata = getData(ncobj, 'min', ij)
sigdata = getData(ncobj, 'sig', ij)

ncobj.close()
fig = ArrayMapFigure()
fig.add(ardata, xgrid, ygrid, 'Bearing AR(1)', [-1, 1], 'AR(1)', map_kwargs)
fig.add(mudata*180./np.pi, xgrid, ygrid, 'Mean bearing', [0, 360.], 'Bearing (degrees)', map_kwargs)
fig.add(mindata, xgrid, ygrid, 'Minimum bearing', [0, 180], 'Bearing (degrees)', map_kwargs)
fig.add(sigdata, xgrid, ygrid, 'Bearing standard deviation', [0, 1], 'Std dev.', map_kwargs)
fig.plot()
#canvas = FC(fig)
#fig
saveFigure(fig, "bearing_stats.png")
Image("bearing_stats.png")



