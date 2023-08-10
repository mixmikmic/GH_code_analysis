get_ipython().magic('pylab inline')
get_ipython().magic('cd Fatiando')

from fatiando.seismic import wavefd

w = wavefd.Ricker(amp=5, cf=10, delay=3.5/10.)
w

import numpy as np
shape = (150, 150)
velocity = 1500*np.ones(shape)
density = 2200*np.ones(shape)

sim = wavefd.ElasticSH(velocity, density, (5,5))                       
sim.add_point_source((shape[0]//2, shape[1]//2), wavefd.Ricker(5, 10.))

sim.run(300)

sim[0]

sim.explore()

#sim.animate(every=5, cutoff=1, blit=True, fps=20, dpi=50, embed=True)
sim.animate(1, embed=True, dpi=50, fps=10, cutoff=0.02, cmap='Greys')

import numpy
from numpy.testing import assert_almost_equal

assert_almost_equal(w(0), 0.0, decimal=3, err_msg="")

fname = sim.cachefile
del sim
reloaded = wavefd.ElasticSH.from_cache(fname)

reloaded

reloaded.run(100)
reloaded

reloaded.animate(embed=True, fps=10, dpi=50)

reloaded.animate(every=5, cutoff=1, blit=True, fps=20, dpi=50, embed=True)

shape = (50, 50)
pvel = 4000*np.ones(shape)
svel = 3000*np.ones(shape)
density = 2200*np.ones(shape)
sim = wavefd.ElasticPSV(pvel, svel, density, spacing=10)
sim.add_point_source((shape[0]//2, shape[1]//2), dip=45, source=wavefd.Ricker(5, 10, 1./10))
sim.cachefile
sim.run(180)

sim

sim.explore()

# Set the parameters of the finite difference grid
shape = (200, 200)
ds = 100.  # spacing
area = [0, shape[0] * ds, 0, shape[1] * ds]
# Set the parameters of the finite difference grid
velocity = np.zeros(shape) + 6000.
velocity[100:, 100:] = 0.
fc = 15.
duration = 2.5
simulation = wavefd.Scalar(velocity, (ds, ds))
simulation.add_point_source((125, 75), wavefd.Gauss(1., fc))

simulation.run(100)

simulation.shape

simulation.velocity.shape

simulation.padding

fig = figure(figsize=(52,5))
ax = fig.add_subplot(111)
fig.colorbar

#simulation.u.shape

fname = simulation.cachefile
del simulation
reloaded = wavefd.Scalar.from_cache(fname)

reloaded.run(400)

reloaded

reloaded.animate(every=5, cutoff=0.05, blit=True, fps=20, dpi=50, embed=True)



