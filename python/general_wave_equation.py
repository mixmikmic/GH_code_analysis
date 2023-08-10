import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from pycav import display

class Wave():

    def __init__(self,s,c,dx,N_x,b,y0,v0):
        self.s = s
        self.c = c
        
        self.N_x = N_x
        self.dx  = dx
        self.x   = dx*np.arange(N_x)
        self.dt  = np.sqrt(s*(dx/c)**2)
        
        self.k0 = -np.pi/(dx)
        self.dk = 2*np.pi/(N_x*dx)
        self.k  = self.k0+self.dk*np.arange(N_x)
        
        self.b = b
        
        self.y0 = y0
        self.v0 = v0
        
        self.y_past = np.zeros_like(y0)
        self.y      = np.zeros_like(y0)
        self.y_new  = np.zeros_like(y0)
        
        self.y_past = y0[:]

    def fft(self):
        self.psi_k = np.fft.fft(self.y)
        self.psi_k = np.fft.fftshift(self.psi_k)
        
    def fixed_boundaries(self):
        self.y[0]  = 0
        self.y[-1] = 0

class DampedWave(Wave):
    
    def __init__(self,s,c,dx,N_x,b,y0,v0):
        Wave.__init__(self,s,c,dx,N_x,b,y0,v0)
        self.y = self.first_step()
        self.fft()
    
    def first_step(self):
        return (self.s/2*(np.roll(self.y0,1)+np.roll(self.y0,-1))+(1-s)*self.y0+(1-self.b*self.dt)*self.dt*self.v0)
    
    def finite_diff(self,n):
        for i in range(n):
            self.y_new = ((self.s*(np.roll(self.y,1)+np.roll(self.y,-1))+2*(1-self.s)*self.y-self.y_past+
                          0.5*self.b*self.dt*self.y_past)/(1+self.b*self.dt/2))
        
            self.y_past = self.y[:]
            self.y      = self.y_new[:]

        self.fft()
        
class StiffWave(Wave):
    
    def __init__(self,s,c,dx,N_x,b,y0,v0):
        Wave.__init__(self,s,c,dx,N_x,b,y0,v0)
        self.a = self.c**2*self.b*self.dt**2/self.dx**4
        self.y = self.first_step()
        self.fft()
    
    def first_step(self):
        return (self.s/2*(np.roll(self.y0,1)+np.roll(self.y0,-1))+self.a/2*(np.roll(self.y0,2)-4*(np.roll(self.y0,1)+
                np.roll(self.y0,-1))+np.roll(self.y0,-2))+(1-self.s+3*self.a)*self.y0+self.dt*self.v0)
    
    def finite_diff(self,n):
        for i in range(n):
            self.y_new = ((self.s*(np.roll(self.y,1)+np.roll(self.y,-1))+2*(1-self.s+3*self.a)*self.y-self.y_past+
                         +self.a*(np.roll(self.y,2)-4*(np.roll(self.y,1)+np.roll(self.y,-1))+np.roll(self.y,-2))))
        
            self.y_past = self.y[:]
            self.y      = self.y_new[:]
        self.fft()    

def gaussian(x,mean,std):
    return np.exp(-(x-mean)**2/(2*std**2))

def vel_gaussian(x,mean,std,c):
    return -c*(x-mean)*gaussian(x,mean,std)/std**2

def gaussian_wp(x,mean,std,freq):
    return np.exp(-(x-mean)**2/(2*std**2))*np.sin(freq*x)

def vel_gaussian_wp(x,mean,std,freq,c):
    return -c*(x-mean)*gaussian_wp(x,mean,std,freq)/std**2+c*freq*gaussian(x,mean,std)*np.cos(freq*x)

def triangle(x,mean,grad):
    tri = np.zeros_like(x)
    idx = np.where(x <= mean)[0][-1]
    tri[:idx] = grad*x[:idx]
    tri[idx:] = -(grad*x[idx])*(x[idx:]-mean)/(x[-1]-mean)+grad*x[idx]
    return tri

def square(x,edges):
    sqr = np.zeros_like(x)
    idx_1 = np.where(x >= edges[0])[0][0]
    idx_2 = np.where(x >= edges[1])[0][0]
    sqr[idx_1:idx_2] = 1.0
    return sqr

dx  = 0.01
N_x = 1001
s   = 0.1
c   = 1.0
b   = 0.5

x = dx*np.arange(N_x)
mu = 9
sigma = 0.4
freq = 10*np.pi
y0 = gaussian_wp(x,mu,sigma,freq)
v0 = vel_gaussian_wp(x,mu,sigma,freq,c)

GaussianWP = DampedWave(s,c,dx,N_x,b,y0,v0)

fig = plt.figure(figsize = (9,9));
ax1 = fig.add_subplot(311);
ax2 = fig.add_subplot(312);
ax3 = fig.add_subplot(313);

line = ax1.plot(GaussianWP.x,GaussianWP.y,'b-')[0];
fft = ax2.plot(GaussianWP.k,np.absolute(GaussianWP.psi_k),'k-')[0];

phase_k_lim = 100
ax3.set_xlim(-phase_k_lim,phase_k_lim)
lower_k = np.where(GaussianWP.k >= -phase_k_lim)[0][0]
upper_k = np.where(GaussianWP.k >= phase_k_lim)[0][0]

phs = ax3.plot(GaussianWP.k[lower_k:upper_k],np.unwrap(np.angle(GaussianWP.psi_k[lower_k:upper_k])),'k-')[0];
ax1.set_ylim(-1.,1.)
ax2.set_xlim(-100,100)

def nextframe(arg):
    line.set_data(GaussianWP.x,GaussianWP.y)
    fft.set_data(GaussianWP.k,np.absolute(GaussianWP.psi_k))
    phs.set_data(GaussianWP.k[lower_k:upper_k],np.unwrap(np.angle(GaussianWP.psi_k[lower_k:upper_k])))
    GaussianWP.finite_diff(10)
    
animate = anim.FuncAnimation(fig,nextframe,frames = 200,interval = 100,repeat = False)

animate = display.create_animation(animate, temp = True)
display.display_animation(animate)

dx  = 0.01
N_x = 1001
s   = 1*10**-4
c   = 1.0
b   = -5*10**-2

x = dx*np.arange(N_x)

mu = 5
sigma = 0.4
freq = -1
y0 = gaussian_wp(x,mu,sigma,freq)
v0 = vel_gaussian_wp(x,mu,sigma,freq,c)

GaussianWP = StiffWave(s,c,dx,N_x,b,y0,v0)

fig = plt.figure(figsize = (9,12));
ax1 = fig.add_subplot(311);
ax2 = fig.add_subplot(312);
ax3 = fig.add_subplot(313);

line = ax1.plot(GaussianWP.x,GaussianWP.y,'b-')[0];
fft = ax2.plot(GaussianWP.k,np.absolute(GaussianWP.psi_k)**2,'k-')[0];

k_lim = 10
ax2.set_xlim(-k_lim,k_lim)
ax3.set_xlim(-k_lim,k_lim)
lower_k = np.where(GaussianWP.k >= -k_lim)[0][0]
upper_k = np.where(GaussianWP.k >= k_lim)[0][0]
angle = np.unwrap(np.angle(GaussianWP.psi_k[lower_k:upper_k]))
phs = ax3.plot(GaussianWP.k[lower_k:upper_k],angle-angle[0],'k-')[0];
ax1.set_ylim(-1.,1.)
ax3.set_ylim(-100,20)

def nextframe(arg):
    line.set_data(GaussianWP.x,GaussianWP.y)
    fft.set_data(GaussianWP.k,np.absolute(GaussianWP.psi_k)**2)
    angle = np.unwrap(np.angle(GaussianWP.psi_k[lower_k:upper_k]))
    phs.set_data(GaussianWP.k[lower_k:upper_k],angle-angle[0])
    GaussianWP.finite_diff(100)
    
animate = anim.FuncAnimation(fig,nextframe,frames = 200,interval = 50,repeat = False)

animate = display.create_animation(animate, temp = True)
display.display_animation(animate)



