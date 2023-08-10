import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import random

get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

def generate_points(n, d, valid_point=lambda p: True):
    """Generates n points each of which lives in d dimensions. Filters points
    by the optional valid_point argument. All points live in the unit cube
    
    Args:
        n: int, number of points to generate
        d: int, number of dimensions of the point
        
    Kwargs:
        valid_point: function, returns bool of whether or not to include the points
        
    Returns:
        list of np.array points
    """
    points = []
    while len(points) < n:
        candidate = np.array([(random.random() - 0.5) * 2 for i in range(d)])
        if valid_point(candidate):
            points.append(candidate)

    return points


def plot_2d_points(points):
    """Plots 2d points
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot([i for (i, j) in points],
            [j for (i, j) in points],
            'bd'
           )
    ax.set_title('Raw Points No Transformation', size='xx-large')
    fig.show()

    
def transform_and_plot_points(points, m, title='My Lovely Points with Transformation'):
    """Transforms the points under matrix m and plots them.
    
    Args:
        points: list of np.array points in a 2d space
        m: np.matrix is a 2x2 matrix to transform the pionts
        
    Kwargs:
        title: str, how to title the plot
        
    Returns:
        None, displays plots in place
    """
    print('Transforming points with matrix:\n', m)
    transformed_points = [m * p.reshape((2, 1)) for p in points]
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for (before, after) in zip(points, transformed_points):
        x = [before[0], after[0]]
        y = [before[1], after[1]]
        ax.plot(x, y, 'y--'
               )
        ax.plot(x[0], y[0], 'gd')
        ax.plot(x[1], y[1], 'rd')
        
    # Trick plt for a legend by replotting last few points
    ax.plot(x[0], y[0], 'gd', label='Before')
    ax.plot(x[1], y[1], 'rd', label='After')
        
    ax.set_title(title, size='xx-large')
    ax.legend(loc='bottom right')
    plt.axes().set_aspect('equal', 'datalim')
    fig.show()

    
def print_eigen_vectors_values(m):
    """
    
    Args:
    
    Returns:
        None, prints in place
    """
    (values, vectors) = LA.eig(m)
    
    print("""
    Eigen Values:
    {values}
    
    Eigen Vectors:
    {vectors}
    """.format(**locals()))

# Generate some points
points = generate_points(10, 2)

# Visualize those points
plot_2d_points(points)

# Create Minor Stretch Transformation
m = np.matrix([[1.1, 0],
               [0, 1.1]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(points, m, 'Minor Stretching of Points')

# Create Minor Stretch Transformation
m = np.matrix([[1.1, 0], [0, 1.1]])
print_eigen_vectors_values(m)

# Create Minor Stretch Transformation
m = np.matrix([[1.1, 0],
               [0, 1.2]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(points, m, 'Minor Stretching of Points with a 1.2')
print_eigen_vectors_values(m)

# Create Minor Stretch Transformation
m = np.matrix([[10, 0],
               [0, 1.1]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(points, m, 'Major Stretching of Points in 1 direction')

# And the corresponding eignvectors and values
print_eigen_vectors_values(m)

# Create Minor Stretch Transformation
m_inv = np.matrix([[1 / 10, 0],
               [0, 1 / 1.1]])

m_inv_la = LA.inv(m)
print(m_inv_la)

#t_points = [m * p.reshape((2, 1)) for p in points]

# Plot How Minot Stretch Transforms Points
#transform_and_plot_points(t_points, m_inv, 'InverseMajor Stretching of Points in 1 direction')

# And the corresponding eignvectors and values
print_eigen_vectors_values(m_inv_la)

# Create Major Stretch Transformation
m = np.matrix([[10, 0], [0, 5]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(points, m, 'Major Stretching of Points in both directions')

# And the corresponding eignvectors and values
print_eigen_vectors_values(m)

# Create Collapse Down to X Transformation
m = np.matrix([[1, 0], [0, 0.0]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(points, m, 'Collapsing Down to X Transformation')
print_eigen_vectors_values(m)

# Create Collapse Down to Y Transformation
m = np.matrix([[0, 0], [0, 1.0]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(points, m, 'Collapsing Down to Y Transformation')
print_eigen_vectors_values(m)

x_points = [np.array([2, 0]),
            np.array([2.3, 0]),
            np.array([-1.2, 0])
           ]

y_points = [np.array([0, 2]),
            np.array([0, 2.3]),
            np.array([0, -1.2])
           ]

# Create Collapse Down to X Transformation
m = np.matrix([[1, 0], [0, 0.0]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(y_points, m, 'Collapsing Down to X Transformation')

# Create Collapse Down to Y Transformation
m = np.matrix([[0, 0], [0, 1.0]])

# Plot How Minot Stretch Transforms Points
transform_and_plot_points(x_points, m, 'Collapsing Down to Y Transformation')

# Create Minor Stretch Transformation
m = np.matrix([[1.1, 0], [0, 1.1]])
transform_and_plot_points(points, m, 'Minor Stretching of Points')

# Create the Inverse of the Minor Stretch Transformation
m_inv = np.matrix([[1/1.1, 0], [0, 1/1.1]])
transform_and_plot_points(points, m, 'Reversing Minor Stretching of Points')

# Rotating Clockwise
import math
theta = math.pi / 4.0

# Rotation Counter Clockwise
m = np.matrix([[math.cos(theta), -1 * math.sin(theta)],
               [math.sin(theta), math.cos(theta)]])
transform_and_plot_points(points, m, 'Rotating Counter Clockwise')

print('Eigen Vectors and Values for Counter Clockwise')
print_eigen_vectors_values(m)

# Rotating Clockwise
theta = -1 * math.pi / 4.0
m = np.matrix([[math.cos(theta), -1 * math.sin(theta)],
               [math.sin(theta), math.cos(theta)]])
transform_and_plot_points(points, m, 'Rotating Clockwise')

print('Eigen Vectors and Values for Clockwise')
print_eigen_vectors_values(m)

#
v = [0.70710678+0.70710678j,  0.70710678-0.70710678j]
print([np.abs(i) for i in v])
print((0.70710678**2 + np.abs(0.70710678j**2))**0.5)

# Rotating Clockwise and Stretching
theta = -1 * math.pi / 4.0
m = np.matrix([[10 * math.cos(theta), -1 * math.sin(theta)],
               [math.sin(theta), math.cos(theta)]])
transform_and_plot_points(points, m, 'Rotating Clockwise')

print('Eigen Vectors and Values for Clockwise')
print_eigen_vectors_values(m)









