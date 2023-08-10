import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

def pendulum(y,t,param):
    theta,omega = y # Unpack the input for easy of use.
    f1 = omega      # Equation one
    f2 = -param[1]/param[0] * np.sin(theta) # Equation 2.
    return([f1,f2]) # Return both functions as a list

# Problem setup
L = 10.0
g = 9.8    # stay down to earth
parameters = [L,g]

# Initial values
theta0 = 30.0*np.pi/180.
omega0 = 0
y0 = [theta0,omega0]

# Setup the time domain for the solution. 
t_end = 30. # 30 seconds. 
N     = 128 
t = np.linspace(0.,t_end,N)

# Now ask SciPy to solve the diffeq for us.
solution = spi.odeint(pendulum,y0,t,args=(parameters,))
solutionT = np.transpose(solution)

fig1 = plt.figure(figsize=(10,7))
plt.title('Pendulum')
plt.xlabel('time (sec)')
plt.ylabel("$\\theta$(deg),$\omega$(deg/s)")
plt.plot(t,solutionT[0]*180./np.pi,label="$\\theta$")
plt.plot(t,solutionT[1]*180./np.pi,label="$\omega$")
plt.legend()
plt.show()
fig2 = plt.figure(figsize=(10,7))
plt.plot(solutionT[0]*180./np.pi,solutionT[1]*180./np.pi)
plt.xlabel("$\\theta$(deg)")
plt.ylabel("$\omega$(deg/s)")
plt.show()

theta0s = np.array([10.,40.,70.,100.,130.,150.,179.,180.],dtype='float')*np.pi/180.

fig2 = plt.figure(figsize=(10,7))
for theta0 in theta0s:
    y0 = [theta0,omega0]
    solution = spi.odeint(pendulum,y0,t,args=(parameters,))
    solutionT = np.transpose(solution)
    plt.plot(solutionT[0]*180./np.pi,solutionT[1]*180./np.pi,label="$\\theta_0$={:7.3f}".format(theta0*180./np.pi))
plt.xlabel("$\\theta$(deg)")
plt.ylabel("$\omega$(deg/s)")
plt.legend()
plt.show()

