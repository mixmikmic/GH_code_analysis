import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
import behalf
from datetime import datetime, timedelta
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import animation, rc
rc('animation', html='jshtml')

data = np.loadtxt('../results/merger_576-12_2000_5000_50/step_0.dat')
pos = data[:,:3]
vel = data[:,3:]
m = np.ones(pos.shape[0]) * (1e5 / 1000)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim((-200, 200))
ax.set_ylim((-200, 200))
data, x, y = np.histogram2d(pos[:,0], pos[:,1], bins=np.linspace(-200, 200, 100))
im = plt.imshow(data.T, interpolation = 'None', origin = 'lower', norm=mpl.colors.LogNorm(), extent=(-200, 200, -200, 200))

t_start = datetime(year=2018, month=5, day=9, hour=8, minute=15)

#Define animation. 
def animate(i) :
    dt = 0.02
    T = i * dt
    X = pos[:,0] + vel[:,0] * T
    Y = pos[:,1] + vel[:,1] * T
    data,x,y = np.histogram2d(X, Y, bins=np.linspace(-200, 200, 100))
    im.set_data(data.T)
    est_time = t_start + timedelta(seconds=28.*i)
    ax.set_title('{:d} Steps, {:.2f} Myr, {:s}'.format(i, T, str(est_time)))
    return im

# call the animator.
anim = animation.FuncAnimation(fig, animate,
                               frames=np.arange(0, 5000, 100), interval=100, blit=False)
anim

data = np.loadtxt('../results/merger_576-12_2000_5000_50/step_0.dat')
pos_full = data[:,:3]
vel_full = data[:,3:]
m_full = np.ones(pos_full.shape[0]) * (1e5 / 1000)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
ax.set_xlim((-100, 100))
ax.set_ylim((-100, 100))
# ax.set_xlabel('x'), ax.set_ylabel('y')
ax.set_xticks([]), ax.set_yticks([])
data, x, y = np.histogram2d(pos_full[:,0], pos_full[:,1], bins=np.linspace(-100, 100, 300))
data[data <= 0.] = 1e-10
print(data.max())
ax.set_facecolor('black')
ax.axis('off')
im = plt.imshow(data.T, interpolation = 'Gaussian', origin = 'lower', cmap='bone', norm=mpl.colors.LogNorm(vmin=0.5, vmax=2, clip=False),  extent=(-200, 200, -200, 200))
plt.tight_layout()

#Define animation. 
def animate_data(i) :
    dt = 0.02
    T = i * dt
    fname = '../results/merger_576-12_2000_5000_50/step_{:d}.dat'.format(i)
    data = np.loadtxt(fname)
    with open(fname, 'r') as f:
        t = f.readlines()[4].split(': ')[-1].split(' ')[-1].split('.')[0]
    X = data[:, 0]
    Y = data[:, 1]
    data,x,y = np.histogram2d(X, Y, bins=np.linspace(-100, 100, 300))
    data[data <= 0.] = 1e-10
    im.set_data(data.T)
    est_time = t_start + timedelta(seconds=28.*i)
#     ax.set_title('{:.2f} Myr'.format(T))
    return im

import os
i_max = 0
for i in range(0, 5000, 10):
    if not os.path.exists('../results/merger_576-12_2000_5000_50/step_{:d}.dat'.format(i)):
        i_max = i - 10
        break
print(i_max)

rc('animation', embed_limit=4*1024*1024)

# call the animator.
anim = animation.FuncAnimation(fig, animate_data,
                               frames=np.arange(0, i_max+1, 10), interval=50, blit=False)
anim

anim.save('merger_final.mp4')



