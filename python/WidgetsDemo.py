from ipywidgets import interact

def times_ten(x):
    return 10 * x

interact(times_ten, x=10);

interact(times_ten, x='(^_^)')

interact(times_ten, x=True)

interact(times_ten, x=(100, 200))

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)

def plot_sine(amplitude, frequency, phase):
    y = amplitude * np.sin(frequency * x - phase)
    plt.plot(x, y)
    plt.ylim(-6, 6)
    
interact(plot_sine,
         amplitude=(0.0, 5.0),
         frequency=(0.1, 10),
         phase=(-5.0, 5.0));

