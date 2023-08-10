#NAME: Foldy-Lax Formulation
#DESCRIPTION: Calculating the wavefield around isotropic scatterers using the Foldy-Lax formulation.

get_ipython().magic('matplotlib inline')
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

#set the incident wavevector
k = [3.0, 0.0, 0.0]
#and wavenumber
K = la.norm(k)

#define a point scatterer
class Scatterer:
    def __init__(self, position, strength):
        self.r = np.array(position)
        self.s = strength
        self.phi = 0.0 #wave amplitude at scatterer
    
#Green's function for 3D Helmholtz
def green(r1, r2):
    return np.exp(1.0j*K*la.norm(r1-r2))/(4.0*np.pi*la.norm(r1-r2))

def incident_wave(r, form):
    #r is a position vector
    incident_amplitude = 0.0j
    if form == 'plane':
        incident_amplitude = np.exp(1.0j*np.dot(k,r))
    elif form == 'spherical':
        incident_amplitude = np.exp(1.0j*K*la.norm(r))/la.norm(r)
    return incident_amplitude

#scattered amplitude from a single scatterer
def scattered_wave(scatterer, r):
    return (scatterer.s)*(scatterer.phi)*green(r,scatterer.r)

def Foldy_Lax(scatterers, incident_form, x, y):
    #x and y are 1D arrays of points at which amplitude is calculated
    #incident_form = 'spherical' or 'plane', form of incident wave
    #scatterers is a list of instances of Scatterer
    
    N = len(scatterers)
    #calculate the amplitude of the incident plane wave at each scatterer
    incident_amplitudes = [incident_wave(scatterers[i].r, incident_form) for i in range(N)]
    
    #calculate the elements of the matrix M (see above)
    M = np.zeros((N,N), dtype = np.complex128)
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i][j] = 1.0
            else:
                M[i][j] = -(scatterers[j].s)*green(scatterers[i].r, scatterers[j].r)
                
    #solve for the amplitudes at each scatterer
    scatterer_amplitudes = la.solve(M, incident_amplitudes)
    for i in range(N):
        scatterers[i].phi = scatterer_amplitudes[i]
        
    #calculate the total wave amplitude at all values of x and y
    amplitudes = np.zeros((y.shape[0], x.shape[0]), dtype = np.complex128)
    for i in range(y.shape[0]):
        for j in range(x.shape[0]):
            r = [x[j],y[i],0.0]
            amplitudes[i][j] = incident_wave(r, incident_form)
            for scatterer in scatterers:
                amplitudes[i][j] += scattered_wave(scatterer, r)
    
    return amplitudes

x = np.linspace(-8,8,80)
y = np.linspace(-8,8,80)

#single point scatterer
position = np.array([0.0,0.0,0.0])
strength = 1.0
p1 = [Scatterer(position, strength)]

#calculate the amplitudes
amplitudes1 = Foldy_Lax(p1, 'plane', x, y)

fig1 = plt.figure(figsize = (10,8))
plt.pcolor(x,y,np.log10(np.absolute(amplitudes1)**2), cmap = 'inferno')
plt.colorbar()
plt.title("Logarithmic intensity for a plane\nwave incident on a single point")
plt.xlabel('x')
plt.ylabel('y')

#create a list of 5 point scatterers in a ring
N2 = 5
p2 = []
for i in range(N2):
    position = np.array([5.0*np.sin(2*np.pi*i/N2), 5.0*np.cos(2*np.pi*i/N2), 0.0])
    strength = 1.0
    p2.append(Scatterer(position, strength))

#calculate the amplitudes
amplitudes2 = Foldy_Lax(p2, 'spherical', x, y)

fig2 = plt.figure(figsize = (10,8))
plt.pcolor(x,y,np.log10(np.absolute(amplitudes2)**2), cmap = 'inferno')
plt.colorbar()
plt.title("Logarithmic intensity around a point\nsource in a ring of 5 scatterers")
plt.xlabel('x')
plt.ylabel('y')

#create a list of 20 point scatterers in a ring
N3 = 20
p3 = []
for i in range(N3):
    position = np.array([5.0*np.sin(2*np.pi*i/N3), 5.0*np.cos(2*np.pi*i/N3), 0.0])
    strength = 1.0
    p3.append(Scatterer(position, strength))

#calculate the amplitudes
amplitudes3 = Foldy_Lax(p3, 'spherical', x, y)

fig3 = plt.figure(figsize = (10,8))
plt.pcolor(x,y,np.log10(np.absolute(amplitudes3)**2), cmap = 'inferno')
plt.colorbar()
plt.title("Logarithmic intensity around a point\nsource in a circle of 20 scatterers")
plt.xlabel('x')
plt.ylabel('y')

#create a list of 20 point scatterers in an ellipse
N4 = 20
p4 = []
for i in range(N4):
    position = np.array([7.0*np.sin(2*np.pi*i/N4), 4.0*np.cos(2*np.pi*i/N4), 0.0])
    strength = 1.0
    p4.append(Scatterer(position, strength))

#calculate the amplitudes
amplitudes4 = Foldy_Lax(p4, 'spherical', x, y)

fig4 = plt.figure(figsize = (10,8))
plt.pcolor(x,y,np.log10(np.absolute(amplitudes4)**2), cmap = 'inferno')
plt.colorbar()
plt.title("Logarithmic intensity around a point\nsource in an ellipse of 20 scatterers")
plt.xlabel('x')
plt.ylabel('y')

