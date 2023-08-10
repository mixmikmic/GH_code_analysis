import numpy as np              # Grab all of the NumPy functions with namespace np

get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

# Import the ODE solver
from scipy.integrate import odeint

def eq_of_motion(w, t, p):
    """
    Defines the differential equations for the forced pendulum system

    Arguments:
        w :  vector of the state variables:
        t :  time
        p :  vector of the parameters:
    """
    theta, theta_dot = w
    g, l, StartTime, T_amp = p

    # Create sysODE = (theta', theta_dot')
    sysODE = [theta_dot,
              -g/l * np.sin(theta) + torque(t, p) / (m * l**2)]
    return sysODE


def torque(t, p):
    """
    defines the torque input to the system
    """
    g, l, StartTime, T_amp = p
    
    # Select one of the two inputs below
    # Be sure to comment out the one you're not using
    
    # Input Option 1: 
    #    Just a step in force beginning at t=DistStart
    # f = F_amp * (t >= DistStart)
    
    # Input Option 2:
    #    A pulse in force beginning at t=StartTime and ending at t=(StartTime + 0.5)
    f = T_amp * (t >= StartTime) * (t <= StartTime + 0.5)
    
    return f

# Define the system parameters
m = 1.0             # mass (kg)
g = 9.81            # acceleration of gravity (m/s^2)
l = 1.0             # length of the pendulum (m)

# Set up simulation parameters

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 10.0
numpoints = 10001

# Create the time samples for the output of the ODE solver
t = np.linspace(0.0, stoptime, numpoints)

# Initial conditions
theta_init = 0.0                        # initial position
theta_dot_init = 0.0                    # initial velocity

# Set up the parameters for the input function
StartTime = 0.5              # Time the f(t) input will begin
T_amp = 2.0                # Amplitude of Disturbance force (N)

# Pack the parameters and initial conditions into arrays 
p = [g, l, StartTime, T_amp]
x0 = [theta_init, theta_dot_init]

# Call the ODE solver.
resp = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Angle (deg)', family='serif', fontsize=22, weight='bold', labelpad=10)

# Plot the first element of resp for all time. It corresponds to the position.
plt.plot(t, resp[:,0] * 180/np.pi, linewidth=2, linestyle = '-', label=r'Response')

# uncomment below and set limits if needed
# plt.xlim(0,5)
# plt.ylim(0,10)

# # Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext,family='serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad = 0.5)

# save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_DirectTorquePendulum_nonlinear.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

def eq_of_motion_linearized(w, t, p):
    """
    Defines the linearized differential equations for the forced pendulum system

    Arguments:
        w :  vector of the state variables:
        t :  time
        p :  vector of the parameters:
    """
    theta, theta_dot = w
    g, l, StartTime, T_amp = p

    # Create sysODE = (theta', theta_dot')
    sysODE = [theta_dot,
              -g/l * theta + torque(t, p) / (m * l**2)]
    return sysODE

# Call the ODE solver.
resp_linearized = odeint(eq_of_motion_linearized, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Angle (deg)', family='serif', fontsize=22, weight='bold', labelpad=10)

# Plot the first element of resp for all time. It corresponds to the position.
plt.plot(t, resp[:,0] * 180/np.pi, linewidth=2, linestyle = '-', label=r'Nonlinear')
plt.plot(t, resp_linearized[:,0] * 180/np.pi, linewidth=2, linestyle = '--', label=r'Linearized')

# uncomment below and set limits if needed
# plt.xlim(0,5)
plt.ylim(-20,25)

# # Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad = 0.5)

# save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_DirectTorquePendulum_LinearizedComparison.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Error (deg)', family='serif', fontsize=22, weight='bold', labelpad=10)

# Plot the first element of resp for all time. It corresponds to the position.
plt.plot(t, (resp[:,0]- resp_linearized[:,0]) * 180/np.pi, linewidth=2, linestyle = '-', label=r'Error')

# uncomment below and set limits if needed
# plt.xlim(0,5)
plt.ylim(-5,5)

# # Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext,family='serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad = 0.5)

# save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE474_DirectTorquePendulum_LinearizedError.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# This cell will just improve the styling of the notebook
from IPython.core.display import HTML
import urllib.request
response = urllib.request.urlopen("https://cl.ly/1B1y452Z1d35")
HTML(response.read().decode("utf-8"))

