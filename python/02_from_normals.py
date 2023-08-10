import numpy as np
from cops_and_robots.robo_tools.fusion.softmax import SoftMax
get_ipython().magic('matplotlib inline')

labels = ['SW', 'NW', 'SE',' NE']
weights = np.array([[-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                   ])

biases = np.array([1, -5, 5, -1,])
pacman = SoftMax(weights, biases, class_labels=labels)
pacman.plot(title='Shifted Pac-Man Bearing Model')

import numpy as np
from cops_and_robots.robo_tools.fusion.softmax import SoftMax
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')

x = np.arange(-5,5,0.1)
y_SESW = float('inf') * x + 2  # Vertical Line
y_SWNW = 0 * x + 3  # Horizontal Line
y_NESE = 0 * x + 3  # Horizontal Line
y_NWNE = float('inf') * x + 2  # Vertical Line
y_NWSE = x + 5
y_NESW = -x + 1

plt.axvline(x = -2, color='pink', ls='-', label="SESW", lw=3)
plt.plot(x, y_SWNW, 'y-', label="SWNW", lw=3)
plt.plot(x, y_NESE, 'g--', label="NESE", lw=3)
plt.axvline(x = -2, color='blue', ls='--' , label="NWNE", lw=3)
plt.plot(x, y_NWSE, 'k-', label="NWSE", lw=3)
plt.plot(x, y_NESW, 'r-', label="NESW", lw=3)

plt.grid()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title('Boundaries Between Intercardinal Spaces')
plt.legend(loc='lower center', bbox_to_anchor=(-0.2, -0.275, 1.4, -0.175),
            mode='expand', ncol=6)

np.set_printoptions(precision=2, suppress=True)
A = np.array([[-1, 1, 0, 0],
              [-1, 0, 1, 0],
              [-1, 0, 0, 1],
              [0, -1, 1, 0],
             ])

# Solve for weights
n = np.array([[2, 0],
              [2, -2],
              [0, -2],
              [0, -2],])

w = np.dot(np.linalg.pinv(A),n)  # Using the Moore-Penrose pseudo-inverse
print('Weights:\n{}'.format(w))

# Solve for biases
d = np.array((4, 10, 6, 6))
biases = np.dot(np.linalg.pinv(A), d)
print('Biases:{}'.format(biases))

labels = ['NE','NW','SW','SE']
pacman = SoftMax(weights=w, biases=biases, class_labels=labels)
pacman.plot(title='Pac-Man Bearing Model')

n = np.array([[0,5,0],
              [0,0,-5],
              [0,-5,0],
              [0,0,5]])
A = np.roll(np.eye(4,4),1,axis=1) - np.eye(4,4)
w = np.dot(np.linalg.pinv(A),n)  # Using the Moore-Penrose pseudo-inverse
print(w)

labels = ['SW','NW','SE','NE']
pacman = SoftMax(weights=w, class_labels=labels)
pacman.plot(title='Pac-Man Bearing Model')

from IPython.core.display import HTML

# Borrowed style from Probabilistic Programming and Bayesian Methods for Hackers
def css_styling():
    styles = open("../styles/custom.css", "r").read()
    return HTML(styles)
css_styling()



