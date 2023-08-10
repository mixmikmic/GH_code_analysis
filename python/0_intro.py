from IPython.display import Image
Image(url="http://www.theprincessplanet.com/comics/2006-04-16.jpg")

Image(url="http://imgs.xkcd.com/comics/python.png")

2 + 5

get_ipython().run_cell_magic('timeit', '', 'x = 0\nfor i in range(10**6):\n    x += 3 * i - 5')

import numpy as np

# press [tab]
np.random.r

get_ipython().magic('matplotlib inline')
import seaborn as sns
from matplotlib import pyplot as plt

X = np.linspace(-5, 5, 500)
for k in [2, 3, 7]:
    Y = np.sin(k * X) / k
    plt.plot(X, Y)

with plt.xkcd():
    X = np.linspace(-5, 5, 500)
    for k in [2, 3, 7]:
        Y = np.sin(k * X) / k
        plt.plot(X, Y)



