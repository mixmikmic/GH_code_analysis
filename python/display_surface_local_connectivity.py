from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *

# configure local connectivity kernel
loc_conn = local_connectivity.LocalConnectivity(cutoff=20.0)
loc_conn.equation.parameters['sigma'] = 10.0
loc_conn.equation.parameters['amp'] = 1.0

# configure cortical surface
ctx = cortex.Cortex(load_default=True)
ctx.local_connectivity = loc_conn
ctx.coupling_strength = 0.0115
ctx.configure()

# plot 
figure()
ax = subplot(111, projection='3d')
x, y, z = ctx.vertices.T
ax.plot_trisurf(x, y, z, triangles=ctx.triangles, alpha=0.1, edgecolor='none')

ctx

