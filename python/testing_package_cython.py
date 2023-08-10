get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import pipe_cython as lb

sim = lb.Pipe_Flow(lx=400, ly=400, tau=5)

sim.run(1000)

sim.u

velocity_magnitude = np.sqrt(sim.u**2 + sim.v**2)
plt.imshow(velocity_magnitude.T)
plt.grid(False)
plt.colorbar()

plt.plot(velocity_magnitude[200, :])

plt.imshow(sim.u)
plt.grid(False)
plt.colorbar()

plt.imshow(sim.v)
plt.grid(False)
plt.colorbar()



