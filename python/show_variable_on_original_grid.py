import sys
sys.path.append("../")

import pyfesom as pf
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
get_ipython().magic('matplotlib notebook')
from matplotlib import cm
from netCDF4 import Dataset

meshpath  ='/csys/nobackup1_CLIDYN/nkolduno/pyfesom/DATA/mesh/COREII'
mesh = pf.load_mesh(meshpath, get3d=True, usepickle=True)

fl = Dataset('/csys/nobackup1_CLIDYN/nkolduno/pyfesom/DATA/results/COREII/fesom.2007.oce.mean.nc')

fl.variables.keys()

fl.variables['temp'].shape

level_data, elem_no_nan = pf.get_data(fl.variables['temp'][0,:], mesh, 500)

map = Basemap(projection='robin',lon_0=0, resolution='c')
x, y = map(mesh.x2, mesh.y2)

get_ipython().run_cell_magic('time', '', 'level_data, elem_no_nan = pf.get_data(fl.variables[\'temp\'][-1,:],mesh,100)\n\nplt.figure(figsize=(10,7))\nmap.drawmapboundary(fill_color=\'0.9\')\nmap.drawcoastlines()\n\nlevels = np.arange(-3., 30., 1)\nplt.tricontourf(x, y, elem_no_nan[::], level_data, levels = levels, cmap=cm.Spectral_r, extend=\'both\')\ncbar = plt.colorbar(orientation=\'horizontal\', pad=0.03);\ncbar.set_label("Temperature, $^{\\circ}$C")\nplt.title(\'Temperature at 100m depth\')\nplt.tight_layout()')

get_ipython().run_cell_magic('time', '', 'level_data, elem_no_nan = pf.get_data(fl.variables[\'temp\'][-1,:],mesh,100)\n\nplt.figure(figsize=(10,7))\nmap.drawmapboundary(fill_color=\'0.9\')\nmap.drawcoastlines()\n\nlevels = np.arange(-3., 20., 1)\nplt.tricontourf(x, y, elem_no_nan[::], level_data, levels = levels, cmap=cm.Spectral_r, extend=\'both\')\ncbar = plt.colorbar(orientation=\'horizontal\', pad=0.03);\ncbar.set_label("Temperature, $^{\\circ}$C")\nplt.title(\'Temperature at 100m depth\')\nplt.tight_layout()')

get_ipython().run_cell_magic('time', '', 'level_data, elem_no_nan = pf.get_data(fl.variables[\'temp\'][-1,:],mesh,100)\n\nplt.figure(figsize=(10,7))\nmap.drawmapboundary(fill_color=\'0.9\')\nmap.drawcoastlines()\n\nlevels = np.arange(-3., 20., 1)\n\neps=(levels.max()-levels.min())/50.\nlevel_data[level_data<=levels.min()]=levels.min()+eps\nlevel_data[level_data>=levels.max()]=levels.max()-eps\nplt.tricontourf(x, y, elem_no_nan[::], level_data, levels = levels, cmap=cm.Spectral_r, extend=\'both\')\ncbar = plt.colorbar(orientation=\'horizontal\', pad=0.03);\ncbar.set_label("Temperature, $^{\\circ}$C")\nplt.title(\'Temperature at 100m depth\')\nplt.tight_layout()')

get_ipython().run_cell_magic('time', '', 'level_data, elem_no_nan = pf.get_data(fl.variables[\'temp\'][-1,:],mesh,100)\n\nplt.figure(figsize=(10,7))\nmap.drawmapboundary(fill_color=\'0.9\')\nmap.drawcoastlines()\n\nplt.tripcolor(x, y, elem_no_nan, \\\n              level_data, \\\n              edgecolors=\'k\',\\\n              lw = 0.01,\n             cmap=cm.Spectral_r,\n             vmin = -3,\n             vmax = 30)\ncbar = plt.colorbar(orientation=\'horizontal\', pad=0.03);\ncbar.set_label("Temperature, $^{\\circ}$C")\nplt.title(\'Temperature at 100m depth\')\nplt.tight_layout()')

m = Basemap(projection='mill',llcrnrlat=45,urcrnrlat=60,            llcrnrlon=-10,urcrnrlon=15,resolution='c')
x2, y2 = m(mesh.x2, mesh.y2)

get_ipython().run_cell_magic('time', '', 'level_data, elem_no_nan = pf.get_data(fl.variables[\'temp\'][-1,:],mesh,0)\n\nplt.figure(figsize=(10,7))\nm.drawmapboundary(fill_color=\'0.9\')\n#m.drawcoastlines()\n\nplt.tripcolor(x2, y2, elem_no_nan, \\\n              level_data, \\\n              edgecolors=\'k\',\\\n              lw = 0.1,\n             cmap=cm.Spectral_r,\n             vmin = 5,\n             vmax = 15)\ncbar = plt.colorbar(orientation=\'horizontal\', pad=0.03);\ncbar.set_label("Temperature, $^{\\circ}$C")\nplt.title(\'Temperature at 0m depth\')\nplt.tight_layout()')



