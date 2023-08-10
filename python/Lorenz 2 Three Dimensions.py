get_ipython().run_line_magic('load_ext', 'base16_mplrc')
get_ipython().run_line_magic('base16_mplrc', 'dark bespin')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from lorenz_rhs import lorenz, get_lorenz_solution

in_0 = [5.0, 5.0, 5.0]
t_max = 20
t_steps = 20000
t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, 
                                            (10.0, 8/3, 28))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solx, soly, solz)
plt.show()

t_max = 1000
t_steps = t_max/0.001

in_1 = [5.0, 5.0, 5.0]
t1, [solx1, soly1, solz1] = get_lorenz_solution(in_1, t_max, t_steps, 
                                            (10.0, 8/3, 80))

in_2 = [5.0001,5.0001,5.0001]
t2, [solx2, soly2, solz2] = get_lorenz_solution(in_2, t_max, t_steps, 
                                            (10.0, 8/3, 80))

from seaborn import xkcd_palette as xkcd
blue, red = xkcd(['denim blue','brick red'])

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solx1[-10000:],soly1[-10000:],solz1[-10000:],color=blue,alpha=0.5)
ax.plot(solx2[-10000:],soly2[-10000:],solz2[-10000:],color=red,alpha=0.5)
plt.show()

fig = plt.figure(figsize=(12,8))
ax1, ax2, ax3, ax4 = [fig.add_subplot(2,2,i+1) for i in range(4)]

ax1.plot(t1[-10000:], solz1[-10000:], color=blue)
ax1.set_title("Z vs. Time, Solution 1")

ax2.plot(soly1[-10000:], solz1[-10000:], color=blue)
ax2.set_title("Y vs. Z, Solution 1")

ax3.plot(t2[-10000:], solz2[-10000:], color=red)
ax3.set_title("Z vs. Time, Solution 2")

ax4.plot(soly2[-10000:], solz2[-10000:], color=red)
ax4.set_title("Y vs. Z, Solution 2")

plt.show()

colors = ['dark green',     'denim blue', 'brick red',   'navy blue',
          'plum',           'olive green',  'deep teal',  'violet red',
          'dark seafoam', 'umber',       'marine',    'dark green',
          'forest green',   'black',   'terracotta',   'dusty purple']

colorz = xkcd(colors)

rz = list(np.linspace(1,100,16))

fig = plt.figure(figsize=(14,14))
axes = [fig.add_subplot(4, 4, i+1, projection='3d') for i in range(len(rz))]

for r, color, ax in zip(rz, colorz, axes):
    t_max = 100
    t_steps = t_max/0.001
    
    in_0 = [8.0, 8.0, 8.0]
    t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, 
                                                (10.0, 8/3, r))
    ax.plot(solx, soly, solz, color=color, alpha=0.5)
    ax.set_title("r = %d"%(r))
            
plt.show()

bz = list(np.linspace(1,16,16))
r = 25

fig = plt.figure(figsize=(14,14))
axes = [fig.add_subplot(4, 4, i+1, projection='3d') for i in range(len(rz))]

for b, color, ax in zip(bz, colorz, axes):
    t_max = 100
    t_steps = t_max/0.001
    
    in_0 = [8.0, 8.0, 8.0]
    t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, 
                                                (10.0, b/3, r))
    ax.plot(solx, soly, solz, color=color, alpha=0.5)
    ax.set_title("b = %d/3"%(b))
            
plt.show()

bz = list(np.linspace(1,16,16))
r = 20
b = 8/3
sigmaz = list(np.logspace(-1,2,16))

fig = plt.figure(figsize=(14,14))
axes = [fig.add_subplot(4, 4, i+1, projection='3d') for i in range(len(rz))]

for sigma, color, ax in zip(sigmaz, colorz, axes):
    t_max = 100
    t_steps = t_max/0.001
    
    in_0 = [8.0, 8.0, 8.0]
    t, [solx, soly, solz] = get_lorenz_solution(in_0, t_max, t_steps, 
                                                (sigma, b, r))
    ax.plot(solx, soly, solz, color=color, alpha=0.5)
    ax.set_title(r"$\sigma = %0.2f$"%(sigma))

plt.show()





