# We'll use the scipy version of the linear algebra
from scipy import linalg


import numpy as np                        # Grab all of the NumPy functions with nickname np

from scipy.integrate import odeint        # We also need to import odeint for the simluations
from scipy import linalg                  # We'll use linalg for the eigenvalue problems

# We want our plots to be displayed inline, not in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

# Define the system as a series of 1st order ODEs (beginnings of state-space form)
def eq_of_motion(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1, x1_dot, x2, x2_dot]
        t :  time
        p :  vector of the parameters:
                  p = [m1, m2, k1, k2, k3, c1, c2, c3]
    """
    x1, x1_dot, x2, x2_dot = w
    m1, m2, k1, k2, k3, c1, c2, c3 = p

    # Create sysODE = (x1', x1_dot', x2', x2_dot'):
    sysODE = [x1_dot,
             (-(k1 + k2) * x1 - (c1 + c2) * x1_dot + k2 * x2 + c2 * x2_dot) / m1,
             x2_dot,
             (k2 * x1 + c2 * x1_dot - (k2 + k3) * x2 - (c2 + c3) * x2_dot) / m2]
              
    return sysODE

# Define the system parameters
m1 = 1.0                # kg
m2 = 2.0                # kg
k1 = 100.0              # N/m
k2 = 50.0               # N/m
k3 = 250.0              # N/m
c1 = 0.8                # Ns/m
c2 = 0.4                # Ns/m
c3 = 0.6                # Ns/m

# Set up simulation parameters 

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 5.0
numpoints = 5001

# Create the time samples for the output of the ODE solver
t = np.linspace(0.0, stoptime, numpoints)

# Now, set up hte intial conditions and call the ODE solver

# Initial conditions
x1_init = 0.5                       # initial x1 position
x1_dot_init = 0.0                   # initial x1 velocity
x2_init = -0.5                      # initial x2 position
x2_dot_init = 0.0                   # initial x2 velocity

# Pack the parameters and initial conditions into arrays 
p = [m1, m2, k1, k2, k3, c1, c2, c3]
x0 = [x1_init, x1_dot_init, x2_init, x2_dot_init]

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

# uncomment below and set limits if needed
# plt.xlim(0,5)
plt.ylim(-0.75,0.75)
plt.yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75], ['', '$-x_0$', '', '$0$', '', '$x_0$', ''])

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext,family='serif',fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('MCHE485_Midterm2_Prob1ci.pdf')

fig.set_size_inches(9,6) # Resize the figure for better display in the notebook

# Define the matrices
M = np.asarray([[m1, 0],
                [0,  m2]])

K = np.asarray([[k1 + k2, -k2],
                [-k2,      k2 + k3]])

eigenvals, eigenvects = linalg.eigh(K, M)

print('\n')
print('The resulting eigenalues are {:.2f} and {:.2f}.'.format(eigenvals[0], eigenvals[1]))
print('\n')
print('So the two natrual frequencies are {:.2f}rad/s and {:.2f}rad/s.'.format(np.sqrt(eigenvals[0]), np.sqrt(eigenvals[1])))
print('\n')

print('\n')
print('The first eigenvector is ' + str(eigenvects[:,0]) + '.')
print('\n')
print('The second eigenvector is ' + str(eigenvects[:,1]) + '.')
print('\n')

# Define a zero damping matrix
c1 = 0.0
c2 = 0.0
c3 = 0.0

C = np.asarray([[c1 + c2, -c2],
                [-c2,      c2 + c3]])



A = np.asarray([[0,                     1,           0,           0],
                [-(k1+k2)/m1, -(c1+c2)/m1,       k2/m1,       c2/m1],
                [0,                     0,           0,           1],
                [k2/m2,             c2/m2, -(k2+k3)/m2, -(c2+c3)/m2]])


eigenvals_ss, eigenvects_ss = linalg.eig(A)

print('\n')
print('The resulting eigenvalues are {:.4}, {:.4}, {:.4}, and {:.4}.'.format(eigenvals_ss[0], eigenvals_ss[1], eigenvals_ss[2], eigenvals_ss[3]))
print('\n')

print('So, the resulting natural frequencies are {:.4}rad/s and {:.4}rad/s.'.format(np.abs(eigenvals_ss[2]), np.abs(eigenvals_ss[0])))
print('\n')

# make 1st entry real
eigvect1_ss = eigenvects_ss[:,0] * np.exp(-1.0j * np.angle(eigenvects_ss[0,0]))
eigvect2_ss = eigenvects_ss[:,2] * np.exp(-1.0j * np.angle(eigenvects_ss[0,2]))

# scale to match the undamped
eigvect1_ss *= (eigenvects[0,0] / eigvect1_ss[0])
eigvect2_ss *= (eigenvects[0,1] / eigvect2_ss[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_ss, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_ss, precision=4, suppress_small=True))
print('\n')

# Form the matrices
A = np.vstack((np.hstack((np.zeros((2,2)),-K)),np.hstack((-K, -C))))

B = np.vstack((np.hstack((-K, np.zeros((2,2)))),np.hstack((np.zeros((2,2)),M))))


# Solve the eigenvalue problem using them
eigenvals_sym, eigenvects_sym = linalg.eig(A, B)

print('\n')
print('The resulting eigenvalues are {:.4}, {:.4}, {:.4}, and {:.4}.'.format(eigenvals_sym[0], eigenvals_sym[1], eigenvals_sym[2], eigenvals_sym[3]))
print('\n')

print('So, the resulting natural frequencies are {:.4}rad/s and {:.4}rad/s.'.format(np.abs(eigenvals_sym[2]), np.abs(eigenvals_sym[0])))
print('\n')

# make 1st entry real
eigvect1_sym = eigenvects_sym[:,0] * np.exp(-1.0j * np.angle(eigenvects_sym[0,0]))
eigvect2_sym = eigenvects_sym[:,2] * np.exp(-1.0j * np.angle(eigenvects_sym[0,2]))

# scale to match the undamped
eigvect1_sym *= (eigenvects[0,0] / eigvect1_sym[0])
eigvect2_sym *= (eigenvects[0,1] / eigvect2_sym[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_sym, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_sym, precision=4, suppress_small=True))
print('\n')

# Define the matrices
m1 = 1.0
m2 = 1.0

k1 = 1.0 
k2 = 1.0
k3 = 1.0

c1 = 0.1
c2 = 0.1
c3 = 0.1

# Redefine the damping matrix
C = np.asarray([[c1 + c2, -c2],
                [-c2,      c2 + c3]])


# Redefine the state-space matrix
A = np.asarray([[0,                     1,           0,           0],
                [-(k1+k2)/m1, -(c1+c2)/m1,       k2/m1,       c2/m1],
                [0,                     0,           0,           1],
                [k2/m2,             c2/m2, -(k2+k3)/m2, -(c2+c3)/m2]])


eigenvals_damped_ss, eigenvects_damped_ss = linalg.eig(A)

print('\n')
print('The resulting eigenvalues are {:.4}, {:.4}, {:.4}, and {:.4}.'.format(eigenvals_damped_ss[0], eigenvals_damped_ss[1], eigenvals_damped_ss[2], eigenvals_damped_ss[3]))
print('\n')

print('So, the resulting natural frequencies are {:.4}rad/s and {:.4}rad/s.'.format(np.abs(eigenvals_damped_ss[2]), np.abs(eigenvals_damped_ss[0])))
print('\n')

# make 1st entry real
eigvect1_damped_ss = eigenvects_damped_ss[:,0] * np.exp(-1.0j * np.angle(eigenvects_damped_ss[0,0]))
eigvect2_damped_ss = eigenvects_damped_ss[:,2] * np.exp(-1.0j * np.angle(eigenvects_damped_ss[0,2]))

# scale to match the undamped
eigvect1_damped_ss *= (eigenvects[0,0] / eigvect1_damped_ss[0])
eigvect2_damped_ss *= (eigenvects[0,1] / eigvect2_damped_ss[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_damped_ss, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_damped_ss, precision=4, suppress_small=True))
print('\n')

# Form the matrices
A = np.vstack((np.hstack((np.zeros((2,2)),-K)),np.hstack((-K, -C))))

B = np.vstack((np.hstack((-K, np.zeros((2,2)))),np.hstack((np.zeros((2,2)),M))))


# Solve the eigenvalue problem using them
eigenvals_damped_sym, eigenvects_damped_sym = linalg.eig(A,B)

# make 1st entry real
eigvect1_damped_sym = eigenvects_damped_sym[:,0] * np.exp(-1.0j * np.angle(eigenvects_damped_sym[0,0]))
eigvect2_damped_sym = eigenvects_damped_sym[:,2] * np.exp(-1.0j * np.angle(eigenvects_damped_sym[0,2]))

# scale to match the undamped
eigvect1_damped_sym *= (eigenvects[0,0] / eigvect1_damped_sym[0])
eigvect2_damped_sym *= (eigenvects[0,1] / eigvect2_damped_sym[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_damped_sym, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_damped_sym, precision=4, suppress_small=True))
print('\n')

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

