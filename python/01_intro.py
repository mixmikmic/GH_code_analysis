# See source at: https://github.com/COHRINT/cops_and_robots/blob/master/src/cops_and_robots/robo_tools/fusion/softmax.py
import numpy as np
from cops_and_robots.robo_tools.fusion.softmax import SoftMax
get_ipython().magic('matplotlib inline')

labels = ['SW', 'NW', 'SE',' NE']
weights = np.array([[-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                   ])
pacman = SoftMax(weights, class_labels=labels)
pacman.plot(title='Unshifted Pac-Man Bearing Model')

biases = np.array([-2.5, 2.5, -2.5, 2.5,])
pacman = SoftMax(weights, biases, class_labels=labels)
pacman.plot(title='Y-Shifted Pac-Man Bearing Model')

biases = np.array([1, -5, 5, -1,])
pacman = SoftMax(weights, biases, class_labels=labels)
pacman.plot(title='Shifted Pac-Man Bearing Model')

weights = np.array([[-10, -10],
                    [-10, 10],
                    [10, -10],
                    [10, 10],
                   ])
biases = np.array([10, -50, 50, -10,])
pacman = SoftMax(weights, biases, class_labels=labels)
pacman.plot(title='Steep Pac-Man Bearing Model')

# Define rotation matrix
theta = np.pi/4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# Rotate weights
weights = np.array([[-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                   ])
weights = np.dot(weights,R)

# Apply rotated biases
biases = np.array([-2 * np.sqrt(2),
                   -3 * np.sqrt(2),
                   3 * np.sqrt(2),
                   2 * np.sqrt(2),])

pacman = SoftMax(weights, biases, class_labels=labels)
pacman.plot(title='Rotated and Shifted Pac-Man Bearing Model')

from IPython.core.display import HTML

# Borrowed style from Probabilistic Programming and Bayesian Methods for Hackers
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

