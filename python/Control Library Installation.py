get_ipython().system(' git clone https://github.com/python-control/python-control.git')
get_ipython().system(' cd python-control && python setup.py install --user')

import control

from IPython.core.display import SVG
SVG(filename='open_loop.svg')

# Transfer function for our plant, with 1 as the numerator and 2s+1 for the denominator
A = 2
B = 1
G_p = control.tf([1], [A, B])

# Print the transfer function
print(G_p)

s = control.tf([1,0],[1])

G_p = 1/(2*s+1)
print(G_p)

import numpy as np

# Array of 20 sample points, from 0 to 20 seconds
T = np.linspace(0,20,20)

# Grab the response
T, Y = control.step_response(G_p, T=T)

# Now let's visualize it
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(T,Y)

# Add legend and labels
fig = plt.figure(figsize=(10,5))
plt.plot(T,Y, '-kx', label='$Y(s)$')
plt.xlabel('Time (sec)')
plt.ylabel('Y(s)')
plt.legend()

import seaborn as sb

# Use the poster style and a whitegrid
plt.style.use('seaborn')
sb.set_context('poster')
sb.set_style('whitegrid')

# Redo our plotting using the new Seaborn style
plt.plot(T,Y)
plt.xlabel('Time (sec)')
plt.ylabel('Y(s)')

# If you don't like it, reset the style using reset_orig()
sb.reset_orig()
plt.plot(T,Y)
plt.xlabel('Time (sec)')
plt.ylabel('Y(s)')

print('Poles: ', control.pole(G_p))
print('Zeros: ', control.zero(G_p))

