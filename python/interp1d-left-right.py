import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

x = np.array([0, 3, 4, 7, 10])
y = np.array([1, -2, 5, 7, 4])
x_small = np.arange(13)-1
x_large = np.arange(1300)/100-1.5

f_previous = interp1d(x, y, kind='previous', bounds_error=False, fill_value='extrapolate')
f_next = interp1d(x, y, kind='next', bounds_error=False, fill_value='extrapolate')
f_nearest = interp1d(x, y, kind='nearest', bounds_error=False, fill_value='extrapolate')

intp = [f_previous, f_next, f_nearest]
cls = ['C1', 'C2', 'C4']
pts = ['x', '+', '.']
lab = ['previous', 'next', 'nearest']

plt.figure(figsize=(12, 8))
plt.plot(x, y, 'C0s', ms=12, label='data')
for i, ip in enumerate(intp):
    plt.plot(x_large, ip(x_large), cls[i]+'--')
    plt.plot(x_small, ip(x_small), cls[i]+pts[i], ms=10, label=lab[i])
plt.legend()
plt.show()

