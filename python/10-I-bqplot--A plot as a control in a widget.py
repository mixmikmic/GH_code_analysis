from __future__ import print_function
import numpy as np

from bqplot import (
    Axis, ColorAxis, LinearScale, DateScale, DateColorScale, OrdinalScale,
    OrdinalColorScale, ColorScale, Scatter, Lines, Figure, Tooltip
)

import ipywidgets as widgets

def fourier_series(amplitudes):
    """
    Compute the fourier sine series given a set of amplitudes. The 
    period of the fundamental of the series is 1 and the series is 
    generated for two periods.
    """
    period = 1.0
    x = np.linspace(0, 2 * period, num=1000)
    y = np.sum(a * np.sin(2 * np.pi * (n + 1) * x / period) 
               for n, a in enumerate(amplitudes))

    return x, y

N_fourier_components = 10
x_data = np.arange(N_fourier_components) + 1
y_data = np.random.uniform(low=-1, high=1, size=N_fourier_components)

# Start by defining a scale for each axis
sc_x = LinearScale()

# The amplitudes are limited to ±1 for this example...
sc_y = LinearScale(min=-1.0, max=1.0)

# You can create a Scatter object without supplying the data at this
# point. It is here so we can see how the control looks.
scat = Scatter(x=x_data, y=y_data, 
               scales={'x': sc_x, 'y': sc_y}, 
               colors=['orange'],
               # This is what makes this plot interactive
               enable_move=True)

# Only allow points to be moved vertically...
scat.restrict_y = True

# Define the axes themselves
ax_x = Axis(scale=sc_x)
ax_y = Axis(scale=sc_y, tick_format='0.2f', orientation='vertical')

# The graph itself...
amplitude_control = Figure(marks=[scat], axes=[ax_x, ax_y], 
                           title='Fourier amplitudes')

# This width is chosen just to make the plot fit nicely with 
# another. Change it if you want.
amplitude_control.layout.width = '400px'

# Let's see what this looks like...
amplitude_control

# Add some test data to make view the result
initial_amplitudes = np.zeros(10)
initial_amplitudes[0] = 1.0

lin_x = LinearScale()
lin_y = LinearScale()

# Note that here, unlike above, we do not set the initial data.
line = Lines(scales={'x': lin_x, 'y': lin_y}, colors=['orange'],
               enable_move=False)

ax_x = Axis(scale=lin_x)
ax_y = Axis(scale=lin_y, tick_format='0.2f', orientation='vertical')

result = Figure(marks=[line], axes=[ax_x, ax_y], 
                title='Fourier sine series',
                # Honestly, I just like the way the animation looks.
                # Value is in milliseconds.
                animation_duration=500)

# Size as you wish...
result.layout.width = '400px'

# Calculate the fourier series....
line.x, line.y = fourier_series(initial_amplitudes)

# Let's take a look!
result

amplitude_control.marks[0].y = initial_amplitudes

# %load solutions/bqplot-as-control/box-widget.py

# %load solutions/bqplot-as-control/update_line.py
def update_line(change):
    pass

# React to changes in the y value....
amplitude_control.marks[0].observe(update_line, names=['y'])

