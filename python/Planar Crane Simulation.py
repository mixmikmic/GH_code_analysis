import numpy as np

# We want out plots to show up inline with the notebook
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

# We need to import the ode solver
from scipy.integrate import odeint

def eq_of_motion(w, t, p):
    """
    Defines the differential equations for the planar pendulum system.

    Arguments:
        w :  vector of the state variables:
        t :  time
        p :  vector of the parameters:
    """
    theta, theta_dot = w
    m, l, Distance, StartTime, Amax, Vmax, Shaper = p

    # Create sysODE = (theta', theta_dot')
    sysODE = [theta_dot,
             -g/l * theta + 1.0/l * x_ddot(t, p)]
    return sysODE


def x_ddot(t, p):
    """
    Defines the accel input to the system.
    
    We'll make a call to our lab function accel_input()
    
    Depending on the desired move distance, max accel, and max velocity, the input is either
    bang-bang or bang-coast-bang
    
    Arguments:
        t : current time step 
        p : vector of parameters
    """
    m, l, Distance, StartTime, Amax, Vmax, Shaper = p
    
    x_ddot = accel_input(Amax,Vmax,Distance,StartTime,t,Shaper)
    
    return x_ddot

# This cell includes functions to generate the acceleartion input

def accel_input(Amax,Vmax,Distance,StartTime,CurrTime,Shaper):
    """
    # Original MATLAB/Octave premable
    ###########################################################################
    # function [accel] = accel_input(Amax,Vmax,Distance,CurrTime,Shaper)
    #
    # Function returns acceleration at a given timestep based on user input
    #
    # Amax = maximum accel, assumed to besymmetric +/-
    # Vmax = maximum velocity, assumed to be symmetric in +/-
    # Distance = desired travel distance 
    # StartTime = Time command should begin
    # CurrTime = current time 
    # Shaper = array of the form [Ti Ai] - matches output format of shaper functions
    #           in toolbox
    #          * If Shaper is empty, then unshaped is run
    #
    #
    # Assumptions:
    #   * +/- maximums are of same amplitude
    #   * command will begin at StartTime (default = 0)
    #   * rest-to-rest bang-coast-bang move (before shaping)
    #
    # Created: 9/23/11 - Joshua Vaughan - vaughanje@gatech.edu
    #
    # Modified: 
    #   10/11/11
    #     * Added hard-coded shaping option - JEV (vaughanje@gatech.edu)
    #
    ###########################################################################
    #
    #
    # Converted to Python on 3/3/13 by Joshua Vaughan (joshua.vaughan@louisiana.edu)
    #
    # Modified:
    #   * 3/26/14 - Joshua Vaughan - joshua.vaughan@louisiana.edu
    #       - Updated some commenting, corrected typos
    #       - Updated numpy import as np
    """

    # These are the times for a bang-coast-bang input 
    t1 = StartTime
    t2 = (Vmax/Amax) + t1
    t3 = (Distance/Vmax) + t1
    t4 = (t2 + t3)-t1
    end_time = t4

    if len(Shaper) == 0:
        # If no shaper is input, create an unshaped command
        if t3 <= t2: # command should be bang-bang, not bang-coast-bang
            t2 = np.sqrt(Distance/Amax)+t1
            t3 = 2.0 * np.sqrt(Distance/Amax)+t1
            end_time = t3
        
            accel = Amax*(CurrTime > t1) - 2*Amax*(CurrTime > t2) + Amax*(CurrTime > t3)
    
        else: # command is bang-coast-bang
            accel = Amax*(CurrTime > t1) - Amax*(CurrTime > t2) - Amax*(CurrTime > t3) + Amax*(CurrTime > t4)

    else: # create a shaped command
        ts = np.zeros((9,1))
        A = np.zeros((9,1))
        # 	Parse Shaper parameters
        for ii in range(len(Shaper)):
            ts[ii] = Shaper[ii,0]  # Shaper impulse times
            A[ii] = Shaper[ii,1]  # Shaper impulse amplitudes

        # Hard-coded for now
        # TODO: be smarter about constructing the total input - JEV - 10/11/11
        accel = (A[0]*(Amax*(CurrTime > (t1+ts[0])) - Amax*(CurrTime > (t2+ts[0])) - Amax*(CurrTime > (t3+ts[0])) + Amax*(CurrTime > (t4+ts[0])))
        +	A[1]*(Amax*(CurrTime > (t1+ts[1])) - Amax*(CurrTime > (t2+ts[1])) - Amax*(CurrTime > (t3+ts[1])) + Amax*(CurrTime > (t4+ts[1])))
        +   A[2]*(Amax*(CurrTime > (t1+ts[2])) - Amax*(CurrTime > (t2+ts[2])) - Amax*(CurrTime > (t3+ts[2])) + Amax*(CurrTime > (t4+ts[2])))
        +   A[3]*(Amax*(CurrTime > (t1+ts[3])) - Amax*(CurrTime > (t2+ts[3])) - Amax*(CurrTime > (t3+ts[3])) + Amax*(CurrTime > (t4+ts[3])))
        +   A[4]*(Amax*(CurrTime > (t1+ts[4])) - Amax*(CurrTime > (t2+ts[4])) - Amax*(CurrTime > (t3+ts[4])) + Amax*(CurrTime > (t4+ts[4])))
        +   A[5]*(Amax*(CurrTime > (t1+ts[5])) - Amax*(CurrTime > (t2+ts[5])) - Amax*(CurrTime > (t3+ts[5])) + Amax*(CurrTime > (t4+ts[5])))
        +   A[6]*(Amax*(CurrTime > (t1+ts[6])) - Amax*(CurrTime > (t2+ts[6])) - Amax*(CurrTime > (t3+ts[6])) + Amax*(CurrTime > (t4+ts[6])))
        +   A[7]*(Amax*(CurrTime > (t1+ts[7])) - Amax*(CurrTime > (t2+ts[7])) - Amax*(CurrTime > (t3+ts[7])) + Amax*(CurrTime > (t4+ts[7])))
        +   A[8]*(Amax*(CurrTime > (t1+ts[8])) - Amax*(CurrTime > (t2+ts[8])) - Amax*(CurrTime > (t3+ts[8])) + Amax*(CurrTime > (t4+ts[8]))))

    return accel

# Define the parameters for simluation
g = 9.81                     # gravity (m/s^2)
l = 4.0                      # cable length (m)

wn = np.sqrt(g / l)            # natural frequency (rad/s)

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 15.0
numpoints = 15001

# Create the time samples for the output of the ODE solver
t = np.linspace(0.,stoptime,numpoints)

# Initial conditions
theta_init = 0.0                        # initial position
theta_dot_init = 0.0                    # initial velocity

# Set up the parameters for the input function
Distance = 2.0              # Desired move distance (m)
Amax = 1.0                  # acceleration limit (m/s^2)
Vmax = 0.35                 # velocity limit (m/s)
StartTime = 0.5             # Time the y(t) input will begin

# Design and define an input Shaper  
Shaper = [] # An empty shaper means no input shaping

# Pack the parameters and initial conditions into arrays 
p = [g, l, Distance, StartTime, Amax, Vmax, Shaper]
x0 = [theta_init, theta_dot_init]

# Call the ODE solver.
response = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)

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
plt.xlabel('Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('Angle (deg)',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(t, response[:,0]*180/np.pi, linewidth=2, label=r'Angle')

# uncomment below and set limits if needed
plt.xlim(0,15)
# plt.ylim(0,10)

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext,family='serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('Crane_AngleResponse.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

def eq_of_motion_withTrolley(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
        t :  time
        p :  vector of the parameters:
    """
    theta, theta_dot, x, x_dot = w
    m, l, Distance, StartTime, Amax, Vmax, Shaper = p

    # Create sysODE = (theta', theta_dot')
    sysODE = [theta_dot,
             -g/l * theta + 1.0/l * x_ddot(t, p),
             x_dot,
             x_ddot(t, p)]
    return sysODE

def x_ddot(t, p):
    """
    Defines the accel input to the system.
    
    We'll make a call to our lab function accel_input()
    
    Depending on the desired move distance, max accel, and max velocity, the input is either
    bang-bang or bang-coast-bang
    
    Arguments:
        t : current time step 
        p : vector of parameters
    """
    m, l, Distance, StartTime, Amax, Vmax, Shaper = p
    
    x_ddot = accel_input(Amax,Vmax,Distance,StartTime,t,Shaper)
    
    return x_ddot

# Define the parameters for simluation
g = 9.81                     # gravity (m/s^2)
l = 4.0                      # cable length (m)

wn = np.sqrt(g / l)            # natural frequency (rad/s)

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 15.0
numpoints = 15001

# Create the time samples for the output of the ODE solver
t = np.linspace(0.0, stoptime, numpoints)

# Initial conditions
theta_init = 0.0                        # initial angle (rad)
theta_dot_init = 0.0                    # initial angular velocity (rad/s)
x_init = 0.0                            # initial trolley position (m)
x_dot_init = 0.0                        # initial trolley velocity (m/s)

# Set up the parameters for the input function
Distance = 2.0              # Desired move distance (m)
Amax = 1.0                  # acceleration limit (m/s^2)
Vmax = 0.35                 # velocity limit (m/s)
StartTime = 0.5             # Time the y(t) input will begin

# Design and define an input Shaper  
Shaper = [] # An empty shaper means no input shaping

# Pack the parameters and initial conditions into arrays 
p = [g, l, Distance, StartTime, Amax, Vmax, Shaper]
x0 = [theta_init, theta_dot_init, x_init, x_dot_init]

# Call the ODE solver.
response_withTrolley = odeint(eq_of_motion_withTrolley, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

theta_resp = response_withTrolley[:,0]
theta_dot_resp = response_withTrolley[:,1]
x_resp = response_withTrolley[:,2]
x_dot_resp = response_withTrolley[:,3]

payload_position = x_resp + l * np.sin(theta_resp)

# Now, let's plot the horizontal position response

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)

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
plt.xlabel('Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('Position (m)',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(t, x_resp, linewidth=2, linestyle='--', label=r'Trolley')
plt.plot(t, payload_position, linewidth=2, linestyle='-', label=r'Payload')

# uncomment below and set limits if needed
plt.xlim(0,15)
plt.ylim(0,3)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('Crane_Position_Response.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = '../styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

