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
    'dx':       1.,     # m
    'dz':       1.,     # m
    'c':        2500.,  # m/s
    'rho':      1000.,  # kg/m^3
    'nx':       100,    # count
    'nz':       200,    # count
    'freq':     2e2,    # Hz
}

nx = systemConfig['nx']
nz = systemConfig['nz']
dx = systemConfig['dx']
dz = systemConfig['dz']

MZ = MiniZephyr(systemConfig)
AH = AnalyticalHelmholtz(systemConfig)
SKS = SparseKaiserSource(systemConfig)

xs, zs = 25, 25
sloc = np.array([xs, zs]).reshape((1,2))

q = SKS(sloc)
uMZ = MZ*q
uAH = AH(sloc)

clip = 100

plotopts = {
    'vmin':     -np.pi,
    'vmax':     np.pi,
    'extent':   [0., dx * nx, dz * nz, 0.],
    'cmap':     cm.bwr,                
}

fig = plt.figure()

ax1 = fig.add_subplot(1,4,1)
plt.imshow(np.angle(uAH.reshape((nz, nx))), **plotopts)
plt.title('AH Phase')

ax2 = fig.add_subplot(1,4,2)
plt.imshow(np.angle(uMZ.reshape((nz, nx))), **plotopts)
plt.title('MZ Phase')

plotopts.update({
    'vmin':     -clip,
    'vmax':     clip,
})

ax3 = fig.add_subplot(1,4,3)
plt.imshow(uAH.reshape((nz, nx)).real, **plotopts)
plt.title('AH Real')

ax4 = fig.add_subplot(1,4,4)
plt.imshow(uMZ.reshape((nz, nx)).real, **plotopts)
plt.title('MZ Real')

fig.tight_layout()

fig = plt.figure()

ax = fig.add_subplot(1,1,1, aspect=0.1)

plt.plot(uAH.real.reshape((nz, nx))[:,xs], label='AnalyticalHelmholtz')
plt.plot(uMZ.real.reshape((nz, nx))[:,xs], label='MiniZephyr')
plt.legend(loc=4)
plt.title('Real part of response through xs=%d'%xs)

uMZr = uMZ.reshape((nz, nx))
uAHr = uAH.reshape((nz, nx))

plotopts.update({
    'cmap': cm.jet,
    'vmin': 0.,
    'vmax': 20.,
})

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
plt.imshow(abs(uAHr - uMZr)/(abs(uAHr)+1e-15) * 100, **plotopts)
cb = plt.colorbar()
cb.set_label('Percent error')

plotopts.update({'vmax': 5.})

ax2 = fig.add_subplot(1,2,2)
plt.imshow(abs(uAHr - uMZr)/(abs(uAHr)+1e-15) * 100, **plotopts)
cb = plt.colorbar()
cb.set_label('Percent error')

fig.tight_layout()



