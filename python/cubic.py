get_ipython().magic('matplotlib inline')

from IPython.html.widgets import interactive
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np

def cubic(a3=1, a2=10, a1=-10, a0=0, xmin=-5, xmax=5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(xmin, xmax, 50)
    y = a3 * x**3 + a2 * x**2 + a1 * x + a0
    ax.axhline(0, c='k', alpha=.5)
    ax.axvline(0, c='k', alpha=.5)
    ax.plot(x, y)
    plt.show()

w = interactive(cubic, a3=(-10., 10.), a2=(-10., 10.), a1=(-10., 10.), a0=(-10., 10), xmin=(-10, 10), xmax=(-10, 10))
display(w)

