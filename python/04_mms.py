import numpy as np
from cops_and_robots.robo_tools.fusion.softmax import SoftMax, make_regular_2D_poly
get_ipython().magic('matplotlib inline')


labels = ['Interior','Exterior','Exterior', 'Exterior', 'Exterior', 'Exterior']
poly = make_regular_2D_poly(5, max_r=2, theta=-np.pi/4, origin=(2,-1))
sm = SoftMax(poly=poly, class_labels=labels, resolution=0.1, steepness=3)
sm.plot(plot_poly=True, plot_normals=False)    

labels = ['Moving Backward', 'Stopped', 'Moving Forward']
sm = SoftMax(weights=np.array([[-2], [0], [2],]),
             biases=np.array([-0.5, 0, -0.5]),
             state_spec='x', class_labels=labels,
             bounds=[-1, 0, 1, 1])
sm.plot()

labels = ['Moving', 'Stopped', 'Moving']
sm = SoftMax(weights=np.array([[-2], [0], [2],]),
             biases=np.array([-0.5, 0, -0.5]),
             state_spec='x', class_labels=labels,
             bounds=[-1, 0, 1, 1])
sm.plot()

labels = ['Moving', 'Stopped', 'Moving']
sm = SoftMax(weights=np.array([[-3], [0], [3],]),
             biases=np.array([-1, 0, -1]),
             state_spec='x', class_labels=labels,
             bounds=[-1, 0, 1, 1])
sm.plot()

from IPython.core.display import HTML

# Borrowed style from Probabilistic Programming and Bayesian Methods for Hackers
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

