import numpy as np              # Grab all of the NumPy functions with "nickname" np

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions
import matplotlib.pyplot as plt

# Set up simulation parameters
t = np.linspace(0, 5, 501)            # Time for simulation, 0-5s with 500 points in-between

# Define the initial conditions x(0) = 1 and x_dot(0) = 0 
x0 = [1.0, 0.0]

# import the IPython widgets
from IPython.html.widgets import interact
from IPython.html import widgets    # Widget definitions
from IPython.display import display # Used to display widgets in the notebook

# Set up the function that plots the repsonse based on slider changes
def plot_response(f = 1.0, z = 0.05):
    # Make the figure pretty, then plot the results
    #   "pretty" parameters selected based on pdf output, not screen output
    #   Many of these setting could also be made default by the .matplotlibrc file
    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)
    plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
    plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True, linestyle=':', color='0.75')
    ax.set_axisbelow(True)
    
    wn = 2 * np.pi * f
    wd = wn * np.sqrt(1 - z**2)
    
    # Define x(t)
    x = np.exp(-z * wn * t) * (x0[0] * np.cos(wd*t) + (z * wn * x0[0] + x0[1])/wd * np.sin(wd*t))
    
    plt.plot(t, x, linewidth=2)
    plt.xlabel('Time (s)',family='serif', fontsize=22, weight='bold', labelpad=5)
    plt.ylabel('Position (m)',family='serif', fontsize=22, weight='bold', labelpad=10)
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 5)
    

# Call the slider interaction
#  f is changes in frequency, allowing between 0.2 and 1.8Hz at 0.1Hz increments
#  z is damping ratio, allowing between 0 and 0.9 and 0.05 increments
interact(plot_response, f=(0.2, 1.8, 0.01), z = (0, 0.9, 0.01))

# Ignore this cell - We just update the CSS to make the notebook look a little bit better

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

