get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
from ipywidgets import *
import numpy as np

@interact(points=['-', '10', '20', '30'])
def render(points=None):
    if points == '-': return
    points = int(points)
    fig, ax = plt.subplots()
    x = np.random.randn(points)
    y = np.random.randn(points)
    ax.scatter(x, y)



