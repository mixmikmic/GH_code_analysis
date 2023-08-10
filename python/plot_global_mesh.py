import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import sys
sys.path.insert(0, "../")
import pydiva2d

Meshdir = '../data/Mesh/Global/'
meshfile = os.path.join(Meshdir, 'fort.22')
meshtopofile = os.path.join(Meshdir, 'fort.23')
figdir = './figures/GlobalMesh/'

GlobalMesh = pydiva2d.Diva2DMesh()
GlobalMesh.read_from(meshfile, meshtopofile)

if not(os.path.exists(figdir)):
    os.makedirs(figdir)

fig = plt.figure()
ax = plt.subplot(111)
GlobalMesh.add_to_plot(linewidth=.1)
plt.savefig(os.path.join(figdir, 'GlobalMesh.png'), dpi=300)
plt.close()

from matplotlib import rcParams
rcParams['agg.path.chunksize'] = 10000

m = Basemap(resolution='l', 
            projection='ortho', 
            lat_0=GlobalMesh.ynode.mean(), 
            lon_0=GlobalMesh.xnode.mean()) 

#m = Basemap(llcrnrlon=GlobalMesh.xnode.min(), llcrnrlat=GlobalMesh.ynode.min(),
#            urcrnrlon=GlobalMesh.xnode.max(), urcrnrlat=GlobalMesh.ynode.max(), 
#            resolution = 'l', epsg=3857)
fig = plt.figure()
#ax = plt.subplot(111)
GlobalMesh.add_to_plot(m=m, linewidth=.1, color=(0.44, 0.55, .83))
#plt.savefig(os.path.join(figdir, 'GlobalMeshBasemap.png'), dpi=300)
plt.show()
plt.close()

lon_init = 0.
lon_end = 3.
lon_step = 1.

for index, lonc in enumerate(np.arange(lon_init, lon_end, lon_step)):
    m = Basemap(resolution='c', projection='ortho',lat_0=30., lon_0=lonc) 
    ax = plt.subplot(111)
    Mesh.add_to_plot(m, linewidth=.1, color=(0.44, 0.55, .83))
    plt.savefig(os.path.join(figdir, "globalmesh{0}.png".format(str(index).zfill(4))), dpi=300)
    plt.close()

GlobalMesh.ynode



