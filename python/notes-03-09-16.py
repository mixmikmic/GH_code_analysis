import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.linalg as la
get_ipython().magic('matplotlib inline')

G = 4*np.pi**2 # Gravitational constant
S1 = [-2,0] # Coordinates of Star 1
S2 = [2,0] # Coordinates of Star 2
M1 = 2 # Mass of Star 1 (in solar mass)
M2 = 1 # Mass of Star 2 (in solar mass)

def f(u,t):
    d1 = la.norm([u[0]-S1[0],u[2]-S1[1]])
    d2 = la.norm([u[0]-S2[0],u[2]-S2[1]])
    dU1dt = u[1]
    dU2dt = -G*M1*(u[0]-S1[0])/d1**3 - G*M2*(u[0]-S2[0])/d2**3
    dU3dt = u[3]
    dU4dt = -G*M1*(u[2]-S1[1])/d1**3 - G*M2*(u[2]-S2[1])/d2**3
    return [ dU1dt , dU2dt , dU3dt , dU4dt ]

u0 = [0,5,3,0] # Initial conditions of the planet: [xposition,xvelocity,yposition,yvelocity]
t = np.linspace(0,30,2000) # Array of time values (in years)
u = spi.odeint(f,u0,t) # Solve system: u = [xposition,xvelocity,yposition,yvelocity]

plt.plot(u[:,0],u[:,2]) # Plot trajectory of the planet
plt.plot(S1[0],S1[1],'ro',markersize=5*M1) # Plot Star 1 as a red star
plt.plot(S2[0],S2[1],'ro',markersize=5*M2) # Plot Star 2 as a red star
plt.axis('equal')
plt.show()

def euler_three_body(S1,S2,M1,M2,u0,tf,numpoints=1000):
    '''
    Plot the trajectory of a planet in Euler's three-body problem.
    
    S1 - list of length 2, coordinates of Star 1
    S2 - list of length 2, coordinates of Star 2
    M1 - mass of Star 1 (in solar mass)
    M2 - mass of Star 2 (in solar mass)
    u0 - list of length 4, initial conditions of the planet: [xposition,xvelocity,yposition,yvelocity]
    tf - final time (in years), plot the trajectory for t in [0,tf]
    numpoints - the number of time values in the plot (default 1000)
    '''
    
    # Define the vector function on the right side of the system of the equations
    def f(u,t):
        G = 4*np.pi**2 # Gravitational constant
        d1 = la.norm([u[0]-S1[0],u[2]-S1[1]]) # Distance from star 1 to planet
        d2 = la.norm([u[0]-S2[0],u[2]-S2[1]]) # Distance from star 2 to planet
        dU1dt = u[1]
        dU2dt = -G*M1*(u[0]-S1[0])/d1**3 - G*M2*(u[0]-S2[0])/d2**3
        dU3dt = u[3]
        dU4dt = -G*M1*(u[2]-S1[1])/d1**3 - G*M2*(u[2]-S2[1])/d2**3
        return [ dU1dt , dU2dt , dU3dt , dU4dt ]

    t = np.linspace(0,tf,numpoints) # Array of time values (in years)
    u = spi.odeint(f,u0,t) # Solve system: u = [xposition,xvelocity,yposition,yvelocity]

    plt.plot(u[:,0],u[:,2]) # Plot trajectory of the planet
    plt.plot(S1[0],S1[1],'ro',markersize=5*M1) # Plot Star 1 as a red star
    plt.plot(S2[0],S2[1],'ro',markersize=5*M2) # Plot Star 2 as a red star
    plt.axis('equal')
    plt.show()

euler_three_body([-1,0],[1,0],2,1,[0,5,3,0],30)

euler_three_body([-1,1],[1,-1],2,1,[0,10,0,5],30)

euler_three_body([-2,0],[2,0],1,1,[0,0,0,5],5)

euler_three_body([-2,0],[2,0],1.5,2.5,[0,4.8,3,0],20)

