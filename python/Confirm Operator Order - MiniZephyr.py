import sys
sys.path.append('../')

import numpy as np
from zephyr.backend import MiniZephyr, SparseKaiserSource, AnalyticalHelmholtz

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

get_ipython().magic('matplotlib inline')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png')
matplotlib.rcParams['savefig.dpi'] = 150 # Change this to adjust figure size

systemConfig = {
    'c':        2000.,  # m/s
    'rho':      1.,     # kg/m^3
    'freq':     10.,    # Hz
}

dx = np.arange(5,15,1)
mxz = 1000.
bound = 250.
sloc = np.array([[500.,500.]])

def getError(dx):
    
    sc = {key: systemConfig[key] for key in systemConfig}
    sc.update({
            'dx': dx,
            'nx': np.floor(mxz / dx),
            'nz': np.floor(mxz / dx),
            'nPML': np.floor(bound / dx)
        })
    
    MZ = MiniZephyr(sc)
    AH = AnalyticalHelmholtz(sc)
    SS = SparseKaiserSource(sc)
    
    q = SS(sloc)
    uMZ = MZ*q
    uAH = AH(sloc)
    uDiff = (uAH - uMZ)#[inside/dx:-inside/dx,inside/dx:-inside/dx]
    sc['error'] = np.sqrt((uDiff * uDiff.conj()).sum()).real
    
    print('nx: %(nx)4d, nz: %(nz)4d, dx: %(dx)8.3f, error: %(error)8.3e'%sc)

    return sc['error']

err = map(getError, dx)

plt.plot(dx, err)
plt.plot(dx, dx**2)
ax = plt.axes()
ax.set_yscale('log')
ax.set_xscale('log')



