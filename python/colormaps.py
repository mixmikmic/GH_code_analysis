from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

from testcard import *

get_ipython().magic('matplotlib inline')

_hesperia_data = np.loadtxt("data/hesperia.csv", delimiter=",", skiprows=4).tolist()
_laguna_data = np.loadtxt("data/laguna.csv", delimiter=",", skiprows=4).tolist()
_lacerta_data = np.loadtxt("data/lacerta.csv", delimiter=",", skiprows=4).tolist()
_mod_plasma_data = np.loadtxt("data/mod_plasma.csv", delimiter=",", skiprows=4).tolist()

one = _hesperia_data
two = _laguna_data

# Concatenate maps "one" and "two", back to back so that the central value
# is de-emphasized, and make the inverse scale "_r" as well:
center_emph = ListedColormap((one[::-1][:192] + two[1:][64:])[10:-10], name="center_emph")
center_emph_r = ListedColormap(center_emph.colors[::-1], name="center_emph_r")
plt.register_cmap(cmap=center_emph)
plt.register_cmap(cmap=center_emph_r)

# Now do the same thing, but to emphasize the central value instead:
center_deemph = ListedColormap((one[:-1][64:] + two[::-1][:192])[20:-20], name="center_deemph")
center_deemph_r = ListedColormap(center_deemph.colors[::-1], name="center_deemph_r")
plt.register_cmap(cmap=center_emph)
plt.register_cmap(cmap=center_emph_r)

for cmap in [center_emph, center_deemph]:
    fig = testcard(cmap)
    fig.savefig("{}.png".format(cmap.name))



