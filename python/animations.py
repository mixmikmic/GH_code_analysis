import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

times = np.linspace(0,1,51)

b.add_dataset('lc', times=times, dataset='lc01')

b.add_dataset('orb', times=times, dataset='orb01')

b.run_compute(irrad_method='none', pbmesh=True)

b.animate()

plt.clf()  # this is necessary since the current axes still has the last frame of the previous animation
b.animate(times=times[:-1:2])

plt.clf()  # this is necessary since the current axes still has the last frame of the previous animation
b.animate('lc01', 'orb01')
# identical to the following:
# b.animate(('lc01', 'orb01'))

plt.clf()  # this is necessary since the current axes still has the last frame of the previous animation
b['lc01@model'].animate()
# identical to the following:
# b.animate('lc01@model')
# b.filter(dataset='lc01', context='model').animate()

plt.clf()  # this is necessary since the current axes still has the last frame of the previous animation
b['lc01@model'].animate(times=times[:-1], uncover=True,                        color='r', linestyle=':',                        highlight_marker='s', highlight_color='g')

plt.clf()  # this is necessary since the current axes still has the last frame of the previous animation
b['pbmesh@model'].animate(times=times[:-1], facecolor='teffs', edgecolor=None)

plt.clf()  # this is necessary since the current axes still has the last frame of the previous animation
b.animate('lc01', 'orb01', uncover=True)

plt.clf()  # this is necessary since the current axes still has the last frame of the previous animation
b.animate({'twig': 'lc01', 'uncover': True}, {'twig': 'orb01'})

plt.clf()
b['lc01@model'].animate(times=times[:-1], uncover=True, fixed_limits=False)

plt.clf()
b['pbmesh@model'].animate(times=times[:-1], fixed_limits=False)

fig = plt.figure()
ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
ax2.set_aspect('equal')

plot1 = {'ax': ax1, 'twig': 'lc01@model'}
plot2 = {'ax': ax2, 'twig': 'pbmesh@model', 'facecolor': 'teffs', 'edgecolor': None}

b.animate(plot1, plot2, times=times[:-1])

