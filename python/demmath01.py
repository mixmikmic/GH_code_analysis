from compecon.tools import nodeunif
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
sns.set_style('dark')
get_ipython().magic('matplotlib notebook')

x = np.linspace(-1, 1, 100)
y = (x + 1) * np.exp(2 * x)
y1 = 1 + 3 * x
y2 = 1 + 3 * x + 8 * x ** 2

plt.figure(figsize=[8, 4])                    
plt.plot(x, y, 'k', linewidth=3, label='Function')
plt.plot(x, y1, 'b', linewidth=3, label='1st order approximation')
plt.plot(x, y2, 'r', linewidth=3, label='2nd order approximation')
plt.title('$y = (x+1) \exp(2x)$ and approximations')
plt.legend(loc='upper left')

nplot = [101, 101]
a = [0, -1]
b = [2, 1]
x1, x2 = nodeunif(nplot, a, b)
x1.shape = nplot
x2.shape = nplot

y = np.exp(x2) * x1 ** 2
y1 = 2 * x1 - x2 - 1
y2 = x1 ** 2 - 2 * x1 * x2 + 0.5 * x2 ** 2 + x2

def newPlot(title, Y, k):
    ax = fig.add_subplot(1, 3, k, projection='3d')
    ax.plot_surface(x1, x2, Y, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.title(title)

fig = plt.figure(figsize=[12, 6])
newPlot('Original function', y, 1)
newPlot('First-order approximation', y1, 2)
newPlot('Second-order approximation', y2, 3)

