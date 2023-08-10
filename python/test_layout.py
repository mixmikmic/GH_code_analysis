get_ipython().magic('matplotlib notebook')

get_ipython().run_cell_magic('html', '', '<style>\n.ui-dialog-titlebar {\n    display: none;\n}\n\n.mpl-message {\n    display: none;\n}\n</style>')

import matplotlib.pyplot as plt
import numpy

x = numpy.random.beta(1, 0.8, 100)
y = numpy.random.beta(0.1, 0.5, 100)

fig, ax = plt.subplots(figsize=(5,3.5))
ax.scatter(x, y)
plt.tight_layout()

from ipywidgets import *
from IPython.display import display

w = IntSlider()
display(w)



