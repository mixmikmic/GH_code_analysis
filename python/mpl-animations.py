get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import xarray as xr

# Define the default parameters values
sigma = 10
rho = 28
beta = 8/3

class L63():
    def __init__(self, sigma, rho, beta, init, dt):
        self.sigma, self.rho, self.beta = sigma, rho, beta 
        self.x, self.y, self.z = init
        self.dt = dt
        self.hist = [init]
    
    def step(self):
        self.x += (self.sigma * (self.y - self.x)) * self.dt
        self.y += (self.x * (self.rho - self.z)) * self.dt
        self.z += (self.x * self.y - self.beta * self.z) * self.dt
        self.hist.append([self.x, self.y, self.z])
    
    def integrate(self, n_steps):
        for n in range(n_steps): self.step()

l1 = L63(sigma, rho, beta, init=[1, 10, 20], dt=1e-2)
l1.integrate(3000)

l2 = L63(sigma, rho, beta, init=[1.1, 10, 20], dt=1e-2)
l2.integrate(3000)

# The hist attribute contains the history for the three variables x, y, and z
np.array(l1.hist).shape

# Grab every tenth time step to speed up plotting
y1 = np.array(l1.hist)[::10, 0]
y2 = np.array(l2.hist)[::10, 0]

# We want to animate something like this
plt.plot(y1);

fig, ax = plt.subplots(figsize=(10, 5));
ax.set_ylim(np.min(y1), np.max(y1))
ax.set_xlim(0, len(y1))
ax.set_title('Time step = 0');

line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)

def animate(i):
    line1.set_data(range(i), y1[:i])
    line2.set_data(range(i), y2[:i])
    ax.set_title(f'Time step = {i}')
    return (line1, line2,)   # Note that the comma is important!

anim = animation.FuncAnimation(fig, animate, frames=300, interval=100, blit=True)

HTML(anim.to_html5_video())

anim.save('tmp.mp4')

def plot_attractor(hists):
    if np.array(hists).ndim == 2: hists = [hists]
    hists = [np.array(h) for h in hists]
    fig = plt.figure(figsize=(10, 10)); ax = fig.gca(projection='3d')
    [ax.plot(h[:,0], h[:,1], h[:,2]) for h in hists]

plot_attractor([l1.hist, l2.hist])

ds = xr.open_dataset('../week4/CAM02.nc', decode_times=False)

ds

ds.time.size

def animate_2D():
    fig, ax = plt.subplots(figsize=(10, 7))
    I = ds['SOLIN'].isel(time=0).plot(ax=ax)
    def animate(i):
        I = ds['SOLIN'].isel(time=i).plot(ax=ax, add_colorbar=False)
        return (I,)
    plt.close()   # Not necessary but will avoid a stray figure popping up
    return animation.FuncAnimation(fig, animate, frames=ds.time.size, interval=150, blit=True)

anim = animate_2D()

HTML(anim.to_html5_video())

HTML(anim.to_jshtml())



