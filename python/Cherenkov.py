from IPython.display import Image
Image(filename="cherenkov.png", width = 700)

import numpy as np
from numpy import linalg as la

#speed of light in medium
cn = 1.0
#charge of particle
q = 1.0

#time discretisation
dt = 0.005
Nsteps = 200
t = [dt*i for i in range(Nsteps)]

#space discretisation
dx = 0.01
N = 100
x = np.linspace(-0.5*N*dx, 0.5*N*dx, N)
y = np.linspace(-0.5*N*dx, 0.5*N*dx, N)

#discretisation error
square_error = dt**2 + 2*(dx/cn)**2

#define particle position in xy plane
def R(T):
    #particle at rest at (0,0) for T<0
    X = 0.0
    Y = 0.0
    if T >= 0:
        X = min(x) + 1.0*T**2
        Y = max(y) - 1.0*T**2
    return np.array([X,Y])

#find the difference between time and space
#interval for obversation and emission.
#off_worldline=0 is the lightcone condition.
def off_worldline(emission_time, position, time):
    return (time - emission_time - la.norm(position - R(emission_time))/cn)**2

#for a given position and time, find the time
#at which the received radiation was emitted
def get_emission_time(position,time):
    emission_time = 0.0
    #check all emission times < time in reverse order
    for T in (t[0:t.index(time)])[::-1]:
        ow = off_worldline(T, position, time)
        if (ow < square_error):
            emission_time = T
            break
    return emission_time

#calculate the scalar potential
def get_phi(position, time):
    T = get_emission_time(position, time)
    return q/(4.0*np.pi*la.norm(position - R(T)))
    
#scalar potential
phi = np.zeros((N,N))
time = t[-1]
it = np.nditer(phi, ['multi_index'])
for element in it:
    i,j = it.multi_index
    position = np.array([x[j],y[i]])
    phi[i,j] = get_phi(position, time)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the electric scalar potential
fig = plt.figure(figsize = (12,9))
plt.pcolor(x,y,np.log10(phi), cmap = 'jet')
plt.xlim(min(x), max(x))
plt.ylim(min(y), max(y))
plt.xlabel('x')
plt.ylabel('y')
plt.title("Electric Scalar Potential of a Moving Point Charge")
plt.colorbar()
plt.show()

