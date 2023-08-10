get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from LB_D2Q9 import pipe_cython as lb

from IPython import display

sim = lb.Pipe_Flow(lx=400, ly=400, tau=5)

total_num_steps = 100
steps_per_image= 1

num_to_record = int(np.ceil(float(total_num_steps)/steps_per_image))
for i in range(num_to_record):
    velocity_magnitude = np.sqrt(sim.u**2 + sim.v**2)
    plt.imshow(velocity_magnitude.T)
    plt.grid(False)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    #plt.pause(10.**-6.)
    
    sim.run(steps_per_image)


sim.run(1)

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



