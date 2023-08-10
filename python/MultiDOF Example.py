import numpy as np

# We'll use the scipy version of the linear algebra
from scipy import linalg

# We'll also use the ode solver to plot the time response
from scipy.integrate import odeint  

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

# Define the matrices
m1 = 10.0
m2 = 1.0
g = 9.81
k = 4 * np.pi**2
l = (m1 * g) / k 
c = 2.0

M = np.asarray([[m1 + m2, -m2 * l],
                [-m2 * l,  m2 * l**2]])

K = np.asarray([[k, 0],
                [0, m2 * g * l]])

eigenvals, eigenvects = linalg.eigh(K,M)

print('\n')
print('The resulting eigenalues are {:.2f} and {:.2f}.'.format(eigenvals[0], eigenvals[1]))
print('\n')
print('So the two natrual frequencies are {:.2f}rad/s and {:.2f}rad/s.'.format(np.sqrt(eigenvals[0]), np.sqrt(eigenvals[1])))
print('\n\n')
print('The first eigenvector is {}.'.format(eigenvects[:,0]))
print('\n')
print('The second eigenvector is {}.'.format(eigenvects[:,1]))
print('\n')

F1 = 1.0
F2 = 0.0

F = [F1, F2]

w = np.linspace(0,6,1200)
X = np.zeros((len(w),2))

# This is (K-w^2 M)^-1 * F
for ii, freq in enumerate(w):
    X[ii,:] = np.dot(linalg.inv(K - freq**2 * M), F)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(X[:,0]) >= 0.5)
X[pos,:] = np.nan
w[pos] = np.nan

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)

# Change the axis units to CMU Serif
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
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('Amplitude',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(w,X[:,0],linewidth=2,label=r'$\bar{x}$')
plt.plot(w,X[:,1],linewidth=2,linestyle="--",label=r'$\bar{\theta}$')

# uncomment below and set limits if needed
# plt.xlim(0,4.5)
plt.ylim(-0.4,0.40)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=16)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# plt.savefig('Spring_Pendulum_Example_Amp.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Plot the magnitude of the response

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)

# Change the axis units to CMU Serif
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
plt.xlabel('Frequency (rad/s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Magnitude', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(w, np.abs(X[:,0]), linewidth=2, label=r'$|\bar{x}|$')
plt.plot(w, np.abs(X[:,1]), linewidth=2, linestyle="--", label=r'$|\bar{\theta}|$')

# uncomment below and set limits if needed
# plt.xlim(0,4.5)
plt.ylim(-0.01, 0.3)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=16)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# plt.savefig('Spring_Pendulum_Example_Mag.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Define the system as a series of 1st order ODES (beginnings of state-space form)
def eq_of_motion(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x, x_dot, theta, theta_dot]
        t :  time
        p :  vector of the parameters:
                  p = [m1, m2, k, c, l, g, wf]
    
    Returns:
        sysODE : An list representing the system of equations of motion as 1st order ODEs
    """
    x, x_dot, theta, theta_dot = w
    m1, m2, k, c, l, g, wf = p

    # Create sysODE = (x', x_dot'):
    sysODE = [x_dot,
              -k/m1 * x - m2/m1 * g * theta + f(t,p),
              theta_dot,
              -k/(m1 * l) * x - (m1 + m2)/m1 * g/l * theta + f(t,p)/l]
    return sysODE



# Define the forcing function
def f(t,p):
    """ 
    Defines the forcing function
    
    Arguments:
        t : time
        p :  vector of the parameters:
             p = [m1, m2, k, l, g, wf]
    
    Returns:
        f : forcing function at current timestep
    """
    
    m1, m2, k, c, l, g, wf = p
    
    # Uncomment below for no force input - use for initial condition response
    #f = 0.0 
    
    # Uncomment below for sinusoidal forcing input at frequency wf rad/s
    f = np.sin(wf * t)
    
    return f

# Set up simulation parameters 

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 100.0
numpoints = 10001

# Create the time samples for the output of the ODE solver
t = np.linspace(0.,stoptime,numpoints)

# Initial conditions
x_init = 0.0                        # initial position
x_dot_init = 0.0                    # initial velocity
theta_init = 0.0                    # initial angle
theta_dot_init = 0.0                # initial angular velocity

wf = np.sqrt(k / m1)                  # forcing function frequency

# Pack the parameters and initial conditions into arrays 
p = [m1, m2, k, c, l, g, wf]
x0 = [x_init, x_dot_init, theta_init, theta_dot_init]

# Call the ODE solver.
resp = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel('Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Position (m \textit{or} rad)',family='serif',fontsize=22,weight='bold',labelpad=10)
# plt.ylim(-1.,1.)

# plot the response
plt.plot(t,resp[:,0], linestyle = '-', linewidth=2, label = '$x$')
plt.plot(t,resp[:,2], linestyle = '--', linewidth=2, label = r'$\theta$')

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=18)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('Spring_Pendulum_Example_TimeResp_Undamped.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Define the system as a series of 1st order ODES (beginnings of state-space form)
def eq_of_motion_damped(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x, x_dot, theta, theta_dot]
        t :  time
        p :  vector of the parameters:
                  p = [m1, m2, k, l, g, wf]
                  
    Returns:
        sysODE : An list representing the system of equations of motion as 1st order ODEs
    """
    x, x_dot, theta, theta_dot = w
    m1, m2, k, c, l, g, wf = p

    # Create sysODE = (x', x_dot'):
    sysODE = [x_dot,
              -k/m1 * x  - c/m1 * x_dot - m2/m1 * g * theta + f(t,p),
              theta_dot,
              -k/(m1 * l) * x - c/(m1 * l) * x_dot - (m1 + m2)/m1 * g/l * theta + f(t,p)/l]
    return sysODE

# Call the ODE solver.
resp_damped = odeint(eq_of_motion_damped, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

# Make the figure pretty, then plot the results
#   "pretty" parameters selected based on pdf output, not screen output
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel('Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Position (m \textit{or} rad)',family='serif',fontsize=22,weight='bold',labelpad=10)
# plt.ylim(-1.,1.)

# plot the response
plt.plot(t,resp_damped[:,0], linestyle = '-', linewidth=2, label = '$x$')
plt.plot(t,resp_damped[:,2], linestyle = '--', linewidth=2, label = r'$\theta$')

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=18)

# If you want to save the figure, uncomment the commands below. 
# The figure will be saved in the same directory as your IPython notebook.
# Save the figure as a high-res pdf in the current folder
# plt.savefig('Spring_Pendulum_Example_TimeResp_Damped.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

