get_ipython().magic('load_ext memory_profiler')
get_ipython().magic('load_ext snakeviz')
get_ipython().magic('load_ext cython')
import holoviews as hv
hv.extension('bokeh','matplotlib')
from IPython.core import debugger
ist = debugger.set_trace

import numpy as np

num = 5000
lmin = -25 #lower simulation box bound in x, y, and z
lmax = +25 #upper simulation box bound in x, y, and z

L = lmax - lmin
box = np.array([L,L,L])
pos = lmin + np.random.random((num,3))*(lmax-lmin)

print('Positions Array Shape:',pos.shape)
print(pos)
print('x min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[0],pos.max(0)[0]))
print('y min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[1],pos.max(0)[1]))
print('z min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[2],pos.max(0)[2]))

get_ipython().run_cell_magic('output', "backend='matplotlib'", 'hv.Scatter3D(pos)')

np.savetxt('data/positions.dat',pos)
np.savetxt('data/box.dat',box)



