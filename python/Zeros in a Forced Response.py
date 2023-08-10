import numpy as np

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the plotting functions 
#  Note: Using the 'from module import *' notation is usually a bad idea. 
import matplotlib.pyplot as plt

# Let's also improve the printing of NumPy arrays.
np.set_printoptions(precision=3, suppress=True)

# Define the matrices
m1 = 1.0
m2 = 1.0
m3 = 1.0
m4 = 1.0

k1 = 4.0 
k2 = 4.0
k3 = 4.0
k4 = 4.0
k5 = 4.0

M = np.array([[m1, 0, 0, 0],
                [0, m2, 0, 0],
                [0, 0, m3, 0],
                [0, 0, 0, m4]])

K = np.array([[k1 + k2, -k2, 0, 0],
                [-k2, k2 + k3, -k3, 0],
                [0, -k3, k3 + k4, -k4],
                [0, 0, -k4, k4+k5]])

# We'll use the scipy version of the linear algebra
from scipy import linalg

eigenvals, eigenvects = linalg.eigh(K,M)

print('\n')
print('The resulting eigenalues are {:.2f}, {:.2f}, {:.2f}, and {:.2f}.'.format(eigenvals[0], eigenvals[1], eigenvals[2], eigenvals[3]))
print('\n')
print('So the natrual frequencies are {:.2f}rad/s, {:.2f}rad/s, {:.2f}rad/s, and {:.2f}rad/s.'.format(np.sqrt(eigenvals[0]), np.sqrt(eigenvals[1]), np.sqrt(eigenvals[2]), np.sqrt(eigenvals[3])))
print('\n')

print('\n')
print('The first eigenvector is ' + str(eigenvects[:,0]) + '.')
print('\n')
print('The second eigenvector is ' + str(eigenvects[:,1]) + '.')
print('\n')
print('The third eigenvector is ' + str(eigenvects[:,2]) + '.')
print('\n')
print('The fourth eigenvector is ' + str(eigenvects[:,3]) + '.')
print('\n')

# Define the equations of motion

# Define the system as a series of 1st order ODEs (beginnings of state-space form)
def eq_of_motion(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1, x1_dot, x2, x2_dot, x3, x3_dot, x4, x4_dot]
        t :  time
        p :  vector of the parameters:
                  p = [m1, m2, m3, m4, k1, k2, k3, k4, k5]
    """
    x1, x1_dot, x2, x2_dot, x3, x3_dot, x4, x4_dot = w
    m1, m2, m3, m4, k1, k2, k3, k4, k5 = p

    # Create sysODE = (x', x_dot'): - Here, we're assuming f(t) = 0
    sysODE = [x1_dot,
             (-(k1+k2)*x1 + k2*x2) / m1,
             x2_dot,
             (k2*x1 - (k2+k3)*x2 + k3*x3) / m2,
             x3_dot,
             (k3*x2 - (k3+k4)*x3 + k4*x4) / m3,
             x4_dot,
             (k4*x3 - (k4+k5)*x4) / m4]
    return sysODE

# Import the ODE solver
from scipy.integrate import odeint  

# Set up simulation parameters 

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 10.0
numpoints = 10001

# Create the time samples for the output of the ODE solver
t = np.linspace(0.0,stoptime,numpoints)

# Initial conditions
x1_init = 0.5                       # initial x1 position
x1_dot_init = 0.0                   # initial x1 velocity
x2_init = 0.5                       # initial x2 position
x2_dot_init = 0.0                   # initial x2 velocity
x3_init = 0.0
x3_dot_init = 0.0
x4_init = 0.0
x4_dot_init = 0.0

# Pack the parameters and initial conditions into arrays 
p = [m1, m2, m3, m4, k1, k2, k3, k4, k5]
x0 = [x1_init, x1_dot_init, x2_init, x2_dot_init, x3_init, x3_dot_init, x4_init, x4_dot_init]

# Call the ODE solver.
resp = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)

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

plt.plot(t,resp[:,0],linewidth=2,label=r'$x_1$')
plt.plot(t,resp[:,2],linewidth=2,linestyle="--",label=r'$x_2$')
plt.plot(t,resp[:,4],linewidth=2,linestyle="-.",label=r'$x_3$')
plt.plot(t,resp[:,6],linewidth=2,linestyle=":",label=r'$x_4$')

# uncomment below and set limits if needed
# plt.xlim(0,5)
plt.ylim(-1,1.35)
plt.yticks([-0.5,0,0.5,1.0],['$-x_0$','$0$','$x_0$','$2x_0$'])

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('FreeVibration_mode_1.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

F1 = 1.0
F2 = 0.0
F3 = 0.0
F4 = 0.0

F = [F1, F2, F3, F4]

w = np.linspace(0,6,1800)
X = np.zeros((len(w),4))

# This is (K-w^2 M)^-1 * F
for ii, freq in enumerate(w):
    X[ii,:] = np.dot(linalg.inv(K - freq**2 * M), F)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(X[:,0]) >= 15)
X[pos,:] = np.nan
w[pos] = np.nan

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(12,8))

plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)

# Change the axis units to CMU Serif
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)


plt.subplot(2,2,1)
plt.plot(w,X[:,0],linewidth=2,label=r'$\bar{x}_1$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_1$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-4,4)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')



plt.subplot(2,2,2)
plt.plot(w,X[:,1],linewidth=2,linestyle="-",label=r'$\bar{x}_2$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_2$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-4,4)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.subplot(2,2,3)
plt.plot(w,X[:,2],linewidth=2,linestyle="-",label=r'$\bar{x}_3$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_3$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-4,4)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.subplot(2,2,4)
plt.plot(w,X[:,3],linewidth=2,linestyle="-",label=r'$\bar{x}_4$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_4$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-4,4)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# # Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext,family='serif',fontsize=16)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5, w_pad=3.0, h_pad=2.0)


# save the figure as a high-res pdf in the current folder
# plt.savefig('Spring_Pendulum_Example_Amp.pdf')

# fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

F1 = 0.0
F2 = 1.0
F3 = 0.0
F4 = 0.0

F = [F1, F2, F3, F4]

w = np.linspace(0,6,1200)
X = np.zeros((len(w),4))

# This is (K-w^2 M)^-1 * F
for ii, freq in enumerate(w):
    X[ii,:] = np.dot(linalg.inv(K - freq**2 * M), F)

# Let's mask the discontinuity, so it isn't plotted
pos = np.where(np.abs(X[:,0]) >= 15)
X[pos,:] = np.nan
w[pos] = np.nan

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(12,8))

plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)

# Change the axis units to CMU Serif
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)


plt.subplot(2,2,1)
plt.plot(w,X[:,0],linewidth=2,label=r'$\bar{x}_1$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_1$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-2,2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


plt.subplot(2,2,2)
plt.plot(w,X[:,1],linewidth=2,linestyle="-",label=r'$\bar{x}_2$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_2$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-2,2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.subplot(2,2,3)
plt.plot(w,X[:,2],linewidth=2,linestyle="-",label=r'$\bar{x}_3$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_3$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-2,2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.subplot(2,2,4)
plt.plot(w,X[:,3],linewidth=2,linestyle="-",label=r'$\bar{x}_4$')
# Define the X and Y axis labels
plt.xlabel('Frequency (rad/s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'$\bar{x}_4$',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.ylim(-2,2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# # Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext,family='serif',fontsize=16)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5, w_pad=3.0, h_pad=2.0)


# save the figure as a high-res pdf in the current folder
# plt.savefig('Spring_Pendulum_Example_Amp.pdf')

# fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

