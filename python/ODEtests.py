get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint # This is the numerical solver

def rhs(Y,t,omega):  # this is the function of the right hand side of the ODE
    y,ydot = Y
    return ydot,-omega*omega*y

t_arr=np.linspace(0,2*np.pi,101)
y_init =[1,0]
omega = 2.0
y_arr=odeint(rhs,y_init,t_arr, args=(omega,))
y,ydot = y_arr[:,0],y_arr[:,1]
plt.ion()
plt.plot(t_arr,y,t_arr,ydot)

# Let's draw a phase portrait, plotting y and ydot together
plt.plot(y,ydot)
plt.title("Solution curve when omega = %4g" % omega)
plt.xlabel("y values")
plt.ylabel("ydot values")

t_arr=np.linspace(0,2*np.pi,101)
y_init =[1,0]
omega = 2.0
y_exact = y_init[0]*np.cos(omega*t_arr) + y_init[1]*np.sin(omega*t_arr)/omega
ydot_exact = -omega*y_init[0]*np.sin(omega*t_arr) + y_init[1]*np.cos(omega*t_arr)
y_arr=odeint(rhs,y_init,t_arr, args=(omega,))
y,ydot = y_arr[:,0],y_arr[:,1]
plt.plot(t_arr,y,t_arr,y_exact)

# We plot the difference
plt.plot(t_arr,y-y_exact)

numsteps=1000001  # adjust this parameter
y_init =[1,0]
omega = 2.0
t_arr=np.linspace(0,2*np.pi,numsteps)
y_exact = y_init[0]*np.cos(omega*t_arr) + y_init[1]*np.sin(omega*t_arr)/omega
ydot_exact = -omega*y_init[0]*np.sin(omega*t_arr) + y_init[1]*np.cos(omega*t_arr)
y_arr=odeint(rhs,y_init,t_arr, args=(omega,))
y,ydot = y_arr[:,0],y_arr[:,1]
plt.plot(t_arr,y-y_exact)

numsteps=100001  # adjust this parameter
y_init =[1,0]
omega = 2.0
t_arr=np.linspace(0,2*1000*np.pi,numsteps)
y_exact = y_init[0]*np.cos(omega*t_arr) + y_init[1]*np.sin(omega*t_arr)/omega
ydot_exact = -omega*y_init[0]*np.sin(omega*t_arr) + y_init[1]*np.cos(omega*t_arr)
y_arr=odeint(rhs,y_init,t_arr, args=(omega,))
y,ydot = y_arr[:,0],y_arr[:,1]
plt.plot(t_arr,y-y_exact)

p1=np.size(t_arr)-1
p0=p1-100
plt.plot(t_arr[p0:p1],y[p0:p1],t_arr[p0:p1],y_exact[p0:p1])

plt.plot(t_arr[p0:p1],y[p0:p1]-y_exact[p0:p1])

plt.plot(t_arr[p0:p1],y_exact[p0:p1],t_arr[p0:p1],3000*(y[p0:p1]-y_exact[p0:p1]))

def rhsSIN(Y,t,omega):  # this is the function of the right hand side of the ODE
    y,ydot = Y
    return ydot,-omega*omega*np.sin(y)

omega = .1   # basic frequency
epsilon = .5 # initial displacement, in radians
t_arr=np.linspace(0,2*100*np.pi,1001)
y_init =[epsilon,0]

# we first set up the exact solution for the linear oscillator
y_exact = y_init[0]*np.cos(omega*t_arr) + y_init[1]*np.sin(omega*t_arr)/omega
ydot_exact = -omega*y_init[0]*np.sin(omega*t_arr) + y_init[1]*np.cos(omega*t_arr)
y_arr=odeint(rhsSIN,y_init,t_arr, args=(omega,))
y,ydot = y_arr[:,0],y_arr[:,1]
plt.ion()
plt.plot(t_arr,y,t_arr,y_exact)

def rhsLZ(u,t,beta,rho,sigma):
    x,y,z = u
    return [sigma*(y-x), rho*x-y-x*z, x*y-beta*z]

sigma = 10.0
beta = 8.0/3.0
rho1 = 29.0
rho2 = 28.8  # two close values for rho give two very different curves

u01=[1.0,1.0,1.0]
u02=[1.0,1.0,1.0]

t=np.linspace(0.0,50.0,10001)
u1=odeint(rhsLZ,u01,t,args=(beta,rho1,sigma))
u2=odeint(rhsLZ,u02,t,args=(beta,rho2,sigma))

x1,y1,z1=u1[:,0],u1[:,1],u1[:,2]
x2,y2,z2=u2[:,0],u2[:,1],u2[:,2]

from mpl_toolkits.mplot3d import Axes3D

plt.ion()
fig=plt.figure()
ax=Axes3D(fig)
ax.plot(x1,y1,z1,'b-')
ax.plot(x2,y2,z2,'r:')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz equations with rho = %g, %g' % (rho1,rho2))

fig=plt.figure()
ax=Axes3D(fig)
ax.plot(x1,y1,z1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig=plt.figure()
ax=Axes3D(fig)
ax.plot(x2,y2,z2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def rhsVDP(y,t,mu):
    return [ y[1], mu*(1-y[0]**2)*y[1] - y[0]]

def jac(y,t,mu):
    return [ [0,1], [-2*mu*y[0]*y[1]-1, mu*(1-y[0]**2)]]

mu=100
t=np.linspace(0,300,10001)
y0=np.array([2.0,0.0])
y,info=odeint(rhsVDP,y0,t,args=(mu,),Dfun=jac,full_output=True)

print("mu = %g, number of Jacobian calls is %d", mu, info['nje'][-1])

plt.plot(t,y)

plt.plot(t,y[:,0])



