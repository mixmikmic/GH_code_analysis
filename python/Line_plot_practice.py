import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

d = ascii.read('d_vs_s.csv', format='csv')
d

x = d['time after microwave pulse']
dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off

fig, ax = plt.subplots()
line1, = ax.plot(x, d['dqc_echo'])

line2, = ax.plot(x, d['sqc_echo'])

ax.legend('dq',loc='lower right')
plt.show()

x = d['time after microwave pulse']
dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off

fig, ax = plt.subplots()
line1, = ax.plot(x, d['dqc_echo'])

line2, = ax.plot(x, d['sqc_echo'])

legend = ['Second-order Phase Cycle','First-order Phase Cycle']
ax.legend(legend,loc='lower right')
ax.set_xlim(100, max(x)-200)
ax.set_xlabel('Time After Microwave Pulse')
ax.set_ylabel('Intensity (Normalized)')
ax.set_ylim((-0.075, 0.075))
ax.set_yticks([])

plt.show()

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy.ma as ma

x = d['time after microwave pulse']
y1 = d['dqc_echo']
y2 = d['sqc_echo']
dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off

mask1 = ma.masked_less(y1, 0.013)
mask2 = ma.masked_less(y2, 0.013)

fig, ax = plt.subplots()
line1, = ax.plot(x, y1,  'grey', label='_nolegend_',linewidth = 3)

line2, = ax.plot(x, y2, 'grey', label='_nolegend_',linewidth = 3)

line3, = ax.plot(x, mask1, 'r', linewidth = 3.2)

line4, = ax.plot(x, mask2, 'b', linewidth = 3.2)

legend = ['2nd-order Transition Echo','1st-order Transition Echo']
ax.legend(legend,loc='lower right')
ax.set_xlim(104, max(x)-200)
ax.set_xlabel('Time After Microwave Pulse (ns)')
ax.set_ylabel('Intensity (Normalized)')
ax.set_ylim((-0.075, 0.075))
ax.set_yticks([])
ax.set_title('Electron Echo Enhancement by Phase Cycle Averaging')

plt.show()

fig.savefig('lineplot_echo.pdf',format='pdf', bbox_inches = 'tight') 



