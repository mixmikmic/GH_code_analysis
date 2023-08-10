import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from findpeaks import findpeaks

n = 80
m = 20
limit = 0
spacing = 3
t = np.linspace(0., 1, n)
x = np.zeros(n)
np.random.seed(0)
phase = 2 * np.pi * np.random.random(m)
for i in range(m):
    x += np.sin(phase[i] + 2 * np.pi * t * i)

peaks = findpeaks(x, spacing=spacing, limit=limit)
plt.plot(t, x)
plt.axhline(limit, color='r')
plt.plot(t[peaks], x[peaks], 'ro')
plt.title('Peaks: minimum value {limit}, minimum spacing {spacing} points'.format(**{'limit': limit, 'spacing': spacing}))
plt.show()

get_ipython().run_cell_magic('timeit', '', 'peaks = findpeaks(x, spacing=100, limit=4.)')

