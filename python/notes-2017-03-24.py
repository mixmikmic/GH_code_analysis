import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
get_ipython().magic('matplotlib inline')

G = 4*np.pi**2 # Gravitational constant
m_star = 1 # Mass of the star
tf = 2 # Final time value, plot the solution for t in [0,tf]
u0 = [1,0,0,7] # Initial conditions: [x position, x speed, y position, y speed]

# Define the right side of system of ODEs
def f(u,t):
    dudt = [0,0,0,0]
    D3 = (u[0]**2 + u[2]**2)**(3/2)
    dudt[0] = u[1]
    dudt[1] = -G*m_star*u[0]/D3
    dudt[2] = u[3]
    dudt[3] = -G*m_star*u[2]/D3
    return dudt

# Compute the solution
t = np.linspace(0,tf,1000)
u = spi.odeint(f,u0,t)

# Plot the trajectory
# The columns of the output are (in order): x, x', y, y'
plt.plot(u[:,0],u[:,2],0,0,'r*')
plt.axis('equal'), plt.grid('on')
plt.show()

# Compute the speed (the norm of the velocity vector)
v = np.sqrt(u[:,1]**2 + u[:,3]**2)
# Compute the distance (the norm of the position vector)
d = np.sqrt(u[:,0]**2 + u[:,2]**2)

plt.plot(t,v,label='Speed')
plt.plot(t,d,label='Distance')
plt.grid('on'), plt.ylim([0,10])
plt.title('The Speed of a Planet in Orbit')
plt.xlabel('Time (years)'), plt.ylabel('Speed (AU/year) / Distance (AU)')
plt.legend()
plt.show()

