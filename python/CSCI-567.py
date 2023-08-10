import numpy as np
import matplotlib.pyplot as plt
import math
get_ipython().magic('matplotlib inline')


x = np.linspace(0.1, 3)

f1x = [(i-1)/i for i in x]
f2x = [math.log(i) for i in x]
f3x = x-1

fig = plt.figure()
ax = fig.gca()

ax.plot(x,f1x, color='r', label='f1x')
ax.plot(x,f2x, color='g', label='f2x')
ax.plot(x,f3x, color='b', label='f3x')
ax.legend([r'$\frac{x-1}{x}$',r'$ln(x)$',r'$x-1$'])
ax.grid(True)



