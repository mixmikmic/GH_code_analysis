#NAME: Wave Equation
#DESCRIPTION: Numerically solving the wave equation in 1D and 2D.

import pycav.pde as pde
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import pycav.display as display

def twoD_gaussian(XX,YY,mean,std):
    return np.exp(-((XX-mean[0])**2+(YY-mean[1])**2)/(2*std**2))

def oneD_gaussian(x,mean,std):
    return np.exp(-((x-mean)**2)/(2*std**2))

def c(x):
    return np.ones_like(x)

def velocity_1d(x,mean,std):
    def V(psi_0):
        return -c(x)*(x-mean)*oneD_gaussian(x,mean,std)/std**2
    return V

def gradient_1d(x,mean,std):
    def D(psi_0):
        return -(x-mean)*oneD_gaussian(x,mean,std)/std**2
    return D

def gradient_2d(x,y,mean,std):
    XX,YY = np.meshgrid(x,y, indexing='ij')
    def D(psi_0):
        dfdx = -(XX-mean[0])*twoD_gaussian(XX,YY,mean,std)/std**2
        dfdy = -(YY-mean[1])*twoD_gaussian(XX,YY,mean,std)/std**2
        return dfdx,dfdy
    return D

def velocity_2d(x,y,mean,std):
    XX,YY = np.meshgrid(x,y, indexing='ij')
    def V(psi_0):
        return -c_2d(x,y)*(x-mean[0])*twoD_gaussian(XX,YY,mean,std)/std**2
    return V

def c_2d(x,y):
    XX,YY = np.meshgrid(x,y, indexing='ij')
    return np.ones_like(XX)

dx = 0.01
x = dx*np.arange(101)

N = 150

mu = 0.75
sigma = 0.05
psi_0_1d = oneD_gaussian(x,mu,sigma)

psi_1d,t = pde.LW_wave_equation(psi_0_1d,x,dx,N,c, 
            init_vel = velocity_1d(x,mu,sigma), init_grad = gradient_1d(x,mu,sigma),
            bound_cond = 'reflective')

fig1 = plt.figure(figsize = (9,6))
line = plt.plot(x,psi_1d[:,0])[0]
plt.ylim([np.min(psi_1d[:,:]),np.max(psi_1d[:,:])])

def nextframe(arg):
    line.set_data(x,psi_1d[:,arg])

animate1 = anim.FuncAnimation(fig1,nextframe, frames = N, interval = 50, repeat = False)
animate1 = display.create_animation(animate1, temp = True)
display.display_animation(animate1)

psi_1d,t = pde.LW_wave_equation(psi_0_1d,x,dx,N,c, 
            init_vel = velocity_1d(x,mu,sigma), init_grad = gradient_1d(x,mu,sigma),
            bound_cond = 'fixed')

fig2 = plt.figure(figsize = (9,6))
line = plt.plot(x,psi_1d[:,0])[0]
plt.ylim([np.min(psi_1d[:,:]),np.max(psi_1d[:,:])])

def nextframe(arg):
    line.set_data(x,psi_1d[:,arg])

animate2 = anim.FuncAnimation(fig2,nextframe, frames = N, interval = 50, repeat = False)
animate2 = display.create_animation(animate2, temp = True)
display.display_animation(animate2)

N = 200
x = dx*np.arange(101)
y = dx*np.arange(201)

XX,YY = np.meshgrid(x,y,indexing='ij')

psi_0_2d = twoD_gaussian(XX,YY,[0.5,0.8],0.1)

psi_2d,t = pde.LW_wave_equation(psi_0_2d,[x,y],dx,2*N,c_2d, a = 0.5,
                init_grad = gradient_2d(x,y,[0.5,0.8],0.1),
                bound_cond = 'reflective')

fig3 = plt.figure(figsize = (9,9))
ax = fig3.gca(projection='3d')
image = ax.plot_surface(XX.T,YY.T,psi_2d[:,:,0].T,cmap = 'plasma')

def nextframe(arg):
    ax.clear()
    ax.set_zlim3d([np.min(2*psi_2d[:,:,:]),np.max(2*psi_2d[:,:,:])])
    ax.plot_surface(XX.T,YY.T,psi_2d[:,:,2*arg].T,cmap = 'plasma')
    
animate3 = anim.FuncAnimation(fig3,nextframe, frames = int(N/2) ,interval = 50, repeat = False)
animate3 = display.create_animation(animate3, temp = True)
display.display_animation(animate3)

def c_2d_variable(x,y):
    XX,YY = np.meshgrid(x,y, indexing='ij')
    return 0.5+0.5*XX

psi_2d,t = pde.LW_wave_equation(psi_0_2d,[x,y],dx,2*N,c_2d_variable, a = 0.5,
                init_grad = gradient_2d(x,y,[0.5,0.8],0.1),  bound_cond = 'reflective')

fig3 = plt.figure(figsize = (9,9))
ax = fig3.gca(projection='3d')
image = ax.plot_surface(XX.T,YY.T,psi_2d[:,:,0].T,cmap = 'plasma')

def nextframe(arg):
    ax.clear()
    ax.set_zlim3d([np.min(2*psi_2d[:,:,:]),np.max(2*psi_2d[:,:,:])])
    ax.plot_surface(XX.T,YY.T,psi_2d[:,:,2*arg].T,cmap = 'plasma')
    
animate4 = anim.FuncAnimation(fig3,nextframe, frames = int(N/2),interval = 50,repeat = False)
animate4 = display.create_animation(animate4, temp = True)
display.display_animation(animate4)



