import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import pi as π

ω = 1

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.gca(projection='3d')

q = np.linspace(-16,16,1000)
t = np.linspace(-4*π, 4*π, 1000)
x = np.cos(ω*t)
y = t
z = np.sin(ω*t)
ax.plot(np.zeros_like(q),np.zeros_like(q),q, color='k', linewidth=1)
ax.plot(q,np.zeros_like(q),np.zeros_like(q), color='k', linewidth=1)
ax.plot(np.zeros_like(q),q,np.zeros_like(q), color='k', linewidth=1)
ax.plot(x, y, z)
ax.text(0,0,18, 'z', fontsize='18')
ax.text(0,18,0, 'y', fontsize='18')
ax.text(18,0,0, 'x', fontsize='18')
ax.view_init(25, 30)
plt.axis('off');

def dBx(t,x,y,z,ω):
    '''x-component of the B field (in units of μ0I)'''
    r = (x-np.cos(ω*t))**2+(y-t)**2+(z-np.sin(ω*t))**2
    return (1/(4.0*π))*(z - np.sin(ω*t) - ω*np.cos(ω*t)*(y-t))/r**3/2

def dBy(t,x,y,z,ω):
    '''y-component of the B field (in units of μ0I)'''
    r = (x-np.cos(ω*t))**2+(y-t)**2+(z-np.sin(ω*t))**2
    return (1/(4.0*π))*(ω*np.sin(ω*t)*(z-np.sin(ω*t))+ω*np.cos(ω*t)*(x-np.cos(ω*t)))/r**3/2

def dBz(t,x,y,z,ω):
    '''z-component of the B field (in units of μ0I)'''
    r = (x-np.cos(ω*t))**2+(y-t)**2+(z-np.sin(ω*t))**2
    return (1/(4.0*π))*(-ω*np.sin(ω*t)*(y-t)-x+np.cos(ω*t))/r**3/2

def trapezoidal_rule(f,x,*params):
    '''The trapezoidal rule for numerical integration of f(x) over x.'''
    
    a,b = x[0],x[-1]
    Δx = x[1] - x[0]
    I = 0
    
    I += 0.5*Δx*(f(a,*params)+f(b,*params))
    for n in range(1,x.size-1):
        I += Δx*f(x[n],*params)
        
    #return Δx*(0.5*(f(a,*params)+f(b,*params)) + np.sum([f(cx,*params) for cx in x[1:-1]]))
    
    return I

N = 500
t = np.linspace(-4*π,4*π,N)

# along the axis
y = np.linspace(-20,20,N)
x,z = 0,0

Bx = np.zeros_like(y)
By = np.zeros_like(y)
Bz = np.zeros_like(y)

for i in range(N):
    Bx[i] = trapezoidal_rule(dBx,t,x,y[i],z,ω)
    By[i] = trapezoidal_rule(dBy,t,x,y[i],z,ω)
    Bz[i] = trapezoidal_rule(dBz,t,x,y[i],z,ω)

plt.plot(y,Bx,label=r'$B_x(0,y,0)$')
plt.plot(y,By, label=r'$B_y(0,y,0)$')
plt.plot(y,Bz, label=r'$B_z(0,y,0)$')

plt.legend(frameon=True, loc='lower right')
plt.xlabel('y')
plt.ylabel(r'$B/\mu_0 I$')



