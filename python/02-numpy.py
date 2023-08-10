get_ipython().magic('load_ext memory_profiler')
get_ipython().magic('load_ext snakeviz')
get_ipython().magic('load_ext cython')
import holoviews as hv
hv.extension('bokeh','matplotlib')
from IPython.core import debugger
ist = debugger.set_trace

import numpy as np
pos = np.loadtxt('data/positions.dat')
box = np.loadtxt('data/box.dat')

print('Read {:d} positions.'.format(pos.shape[0]))
print('x min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[0],pos.max(0)[0]))
print('y min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[1],pos.max(0)[1]))
print('z min/max: {:+4.2f}/{:+4.2f}'.format(pos.min(0)[2],pos.max(0)[2]))

import numpy as np

def potentialEnergyFunk(r,width=1.0,height=10.0):
    '''
    Calculates the (soft) potential energy between two atoms
    
    Parameters
    ----------
    r: ndarray (float)
        separation distances between two atoms
    height: float
        breadth of the potential i.e. where the potential goes to zero
    width: float
        strength/height of the potential
    '''
    U = np.zeros_like(r)
    mask = (r<width) #only do calculation below the cutoff width
    U[mask] = 0.5 * height * (1 + np.cos(np.pi*r[mask]/width))
    return U

get_ipython().run_cell_magic('opts', 'Curve [width=600,show_grid=True,height=350]', "\ndr = 0.05          # spacing of r points\nrmax = 10.0        # maximum r value\npts = int(rmax/dr) # number of r points\nr = np.arange(dr,rmax,dr)\n\ndef plotFunk(width,height,label='dynamic'):\n    U = potentialEnergyFunk(r,width,height)\n    return hv.Curve((r,U),kdims=['Separation Distance'],vdims=['Potential Energy'],label=label)\n    \ndmap = hv.DynamicMap(plotFunk,kdims=['width','height'])\ndmap = dmap.redim.range(width=((1.0,10.0)),height=((1.0,5.0)))\ndmap*plotFunk(10.0,5.0,label='width: 10., height: 5.')*plotFunk(1.0,1.0,label='width: 1., height: 1.')")

from math import sqrt

def calcTotalEnergy1(pos,box):
    '''
    Parameters
    ----------
    pos: ndarray, size (N,3), (float)
        array of cartesian coordinate positions
    
    box: ndarray, size (3), (float)
        simulation box dimensions
    '''
    
    #sanity check
    assert box.shape[0] == 3
    
    # This next line is rather unpythonic but essentially it convinces
    # numpy to perform a subtraction between the full Cartesian Product
    # of the positions array
    dr = np.abs(pos - pos[:,np.newaxis,:])
    
    #still need to apply periodic boundary conditions
    dr = np.where(dr>box/2.0,dr-box,dr)
        
    dist = np.sqrt(np.sum(np.square(dr),axis=-1))
    
    # calculate the full N x N distance matrix
    U = potentialEnergyFunk(dist)

    # extract the upper triangle from U
    U = np.triu(U,k=1) 
    
    return U.sum()  

get_ipython().run_cell_magic('prun', '-D prof/numpy1.prof', 'energy = calcTotalEnergy1(pos,box)')

with open('energy/numpy1.dat','w') as f:
    f.write('{}\n'.format(energy))

memprof = get_ipython().magic('memit -o calcTotalEnergy1(pos,box)')

usage = memprof.mem_usage[0]
incr = memprof.mem_usage[0] - memprof.baseline
with open('prof/numpy1.memprof','w') as f:
    f.write('{}\n{}\n'.format(usage,incr))

from math import sqrt

def calcTotalEnergy2(pos,box):
    '''
    Parameters
    ----------
    pos: ndarray, size (N,3), (float)
        array of cartesian coordinate positions
    
    box: ndarray, size (3), (float)
        simulation box dimensions
    '''
    
    #sanity check
    assert box.shape[0] == 3
    
    # This next line is rather unpythonic but essentially it convinces
    # numpy to perform a subtraction between the full Cartesian Product
    # of the positions array
    dr = np.abs(pos - pos[:,np.newaxis,:])
    
    #extract out upper triangle
    dr = dr[np.triu_indices(dr.shape[0],k=1)]  #<<<<<<<
    
    #still need to apply periodic boundary conditions
    dr = np.where(dr>box/2.0,dr-box,dr)
        
    dist = np.sqrt(np.sum(np.square(dr),axis=-1))
    
    # calculate the full N x N distance matrix
    U = potentialEnergyFunk(dist)
    
    return U.sum()  

get_ipython().run_cell_magic('prun', '-D prof/numpy2.prof', 'energy = calcTotalEnergy2(pos,box)')

with open('energy/numpy2.dat','w') as f:
    f.write('{}\n'.format(energy))

memprof = get_ipython().magic('memit -o calcTotalEnergy2(pos,box)')

usage = memprof.mem_usage[0]
incr = memprof.mem_usage[0] - memprof.baseline
with open('prof/numpy2.memprof','w') as f:
    f.write('{}\n{}\n'.format(usage,incr))



