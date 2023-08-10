# Grab all of the NumPy functions with namespace np
import numpy as np              

get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

import control # This will import the control library.  

x0 = 0.15    # The initial condition (m)

# Define the natural frequency. We use the numpy sqrt function, so we need to preface its call with np.
w_n = np.sqrt(2)    

# Define the damping ratio. 
zeta = 1 / (2 * np.sqrt(2))

# We'll also define the mass of the system
m = 1.0     # mass (kg)

get_ipython().run_line_magic('pinfo', 'np.linspace')

t = np.linspace(0, 10, 1001) # define a time vector of 1001 equally-spaced points between 0 and 10

w_d = w_n * np.sqrt(1 - zeta**2)  # Damped ratural freq (rad/s)
theta = np.arccos(zeta)           # Phase shift (rad)

x = (x0 / np.sqrt(1 - zeta**2)) * np.exp(-zeta * w_n * t) * np.sin(w_d * t + theta)

plt.plot(t, x)

upper_decay_env = (x0 / np.sqrt(1 - zeta**2)) * np.exp(-zeta * w_n * t)
lower_decay_env = -upper_decay_env # It's just the opposite

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

# Remove the top and right axis borders
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Position (m)', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(t, x, linewidth=2, linestyle='-', label=r'Response')

# Now add the decay envolopes with a low zorder to push them behind the response
plt.plot(t, upper_decay_env, linewidth=2, linestyle='-.', zorder=1, label=r'Upper Env.')
plt.plot(t, lower_decay_env, linewidth=2, linestyle='--', zorder=1, label=r'Lower Env.')

# uncomment below and set limits if needed
# plt.xlim(0, 5)
# plt.ylim(0, 10)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_mass_spring_damper_response.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# Define the numerator and denominator of the transfer function
num = [1 / m]
den = [1, 2 * zeta * w_n, w_n**2]

# Define the transfer function form of the system defined by num and den
sys = control.tf(num, den)

print(sys)

sys.pole()

sys.zero()

poles, zeros = control.pzmap.pzmap(sys)

# Define the initial conditions for the system.
# Here they are the initial velocity and position of the mass
sys_init = [0, 0.15]      

time_out, response = control.initial_response(sys, t, sys_init)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Position (m)', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(time_out, response, linewidth=2, linestyle='-', label=r'Response')

# uncomment below and set limits if needed
# plt.xlim(0, 5)
# plt.ylim(0, 10)

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_initial_response.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

time_imp, impulse_response = control.impulse_response(sys, t)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Position (m)', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(time_imp, impulse_response, linewidth=2, linestyle='-', label=r'Response')


# uncomment below and set limits if needed
# plt.xlim(0, 5)
# plt.ylim(0, 10)

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_impulse_response.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

time_step, step_response = control.step_response(sys, t)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Position (m)', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(time_step, step_response, linewidth=2, linestyle='-', label=r'Response')

# uncomment below and set limits if needed
# plt.xlim(0, 5)
# plt.ylim(0, 10)

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_step_response.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# This cell will just improve the styling of the notebook
from IPython.core.display import HTML
import urllib.request
response = urllib.request.urlopen("https://cl.ly/1B1y452Z1d35")
HTML(response.read().decode("utf-8"))

