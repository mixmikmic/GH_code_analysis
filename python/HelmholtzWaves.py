#NAME: Waves in the Steady State
#DESCRIPTION: Solving the Helmholtz equation to find the electric field strength in the steady state.

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

#kronecker delta function
def kd(i,j):
    d = 0.0
    if i == j:
        d = 1.0
    return d

#the array is NxN
N = 101
#grid element length & width/m
D = 0.2
#wavelength of radiation/m
wavelength = 3.2
#wavenumber
wavenumber = 2*np.pi/wavelength

#refractive index array
n = np.ones((N,N), dtype = np.complex128)
for i in range(N):
    for j in range(N):
        if np.sqrt((D*(i-N/2))**2 + (D*(j-N/2))**2) > 8.0:
            n[i,j] = 2.2 - 0.1j

B = np.zeros((2*N+1,N*N), dtype = np.complex128)
for p in range(N*N):
    for q in range(p-N, p+N+1):
        if N*N > q and 0 <= q:
            i,j,k,l = int((p-(p%N))/N), int(p%N), int((q-(q%N))/N), int(q%N) #indices of the NxNxNxN array
            term1 = (1.0/D**2)*(kd(i+1,k)*kd(j,l)+kd(i-1,k)*kd(j,l)-2*kd(i,k)*kd(j,l))
            term2 = (1.0/D**2)*(kd(i,k)*kd(j+1,l)+kd(i,k)*kd(j-1,l)-2*kd(i,k)*kd(j,l))
            term3 = kd(i,k)*kd(j,l)*(wavenumber/n[i,j])**2
            B[N+p-q,q] = term1 + term2 + term3

#create flattened source matrix            
Sf = np.zeros((N*N), dtype = np.complex128)
for i in range(N):
    for j in range(N):
        Sf[N*i+j] = kd(int(N/4),i)*kd(int(N/2),j)
        
#find flattened E matrix
Ef = la.solve_banded((N,N),B,Sf)

#unflatten E matrix
E = np.zeros((N,N), dtype = np.complex128)
S = np.zeros((N,N), dtype = np.complex128)
for i in range(N):
    for j in range(N):
        E[i,j] = Ef[N*i+j]
        S[i,j] = Sf[N*i+j]

#create x and y values for plot
x = np.linspace(0.0, D*N, N)
y = np.linspace(0.0, D*N, N)

#generate plot
plt.figure(figsize = (10,8))
plt.pcolor(x,y,np.absolute(E), cmap = "inferno")#, extent = (x.min(), x.max(), y.min(), y.max()), interpolation = 'none')
plt.colorbar()
#add black contours for refractive index
plt.contour(x,y, np.absolute(n), colors = 'k')
#add white contours for the source
plt.contour(x,y,np.absolute(S), colors = 'w')
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.title("Steady State Electric Field Strength")
plt.show()

