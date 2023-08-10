# Import Python modules and 
import numpy as np
from numpy import zeros, ones, linspace, arange, array
from numpy.linalg import inv

import sympy as sp
from sympy.plotting import plot3d
from sympy import symbols, solve, diff, Matrix

from pprint import pprint

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib as mpl
# printing and plotting settings 
sp.init_printing(use_latex='mathjax')
get_ipython().magic('matplotlib inline') # inline plotting

#mpl.rcParams['figure.figsize'] = (12,7)
#mpl.rcParams['font.size'] = 14
#mpl.rcParams['legend.fontsize'] = 14

# import mechpy modules
from composites import Qf, T1, T2, T1s, T2s

# Define Material Properties and compute ABD matrix

plythk = 0.0025
plyangle = array([0,45]) * np.pi/180 # angle for each ply  # [0,90,-45,45,0][0 45 -45 90 0]
nply = len(plyangle) # number of plies
laminatethk = zeros(nply) + plythk
H =   sum(laminatethk) # plate thickness
# Create z dimensions of laminate
z_ = zeros(nply+1); z_[0] = -H/2
zmid_ = zeros(nply)
for i in range(nply):
    z_[i+1] = z_[i] + laminatethk[i]
    zmid_[i] = z_[i] + laminatethk[i]/2

a_ =   20  # plate width;
b_ =   10  # plate height
q_ = -5.7 # plate load;
# Transversly isotropic material properties
E1 = 150e9
E2 = 12.1e9
nu12 = 0.248
G12 = 4.4e9
nu23 = 0.458
G23 = E2 / (2*(1+nu23))
# Failure Strengths
F1t =  1500e6
F1c = -1250e6
F2t =  50e6
F2c = -200e6
F12t =  100e6
F12c =  -100e6
Strength = array([[F1t, F1c],
                    [F2t, F2c],
                    [F12t, F12c]])

A = zeros((3,3)); B = zeros((3,3)); D = zeros((3,3))  
Q = Qf(E1, E2, nu12, G12 )
for i in range(nply):  # = nply
    Qbar = inv(T1(plyangle[i])) @ Q   @ T2(plyangle[i]) # solve(T1(plyangle[i]), Q) @ T2(plyangle[i])
    A += Qbar*(z_[i+1]-z_[i])
    # coupling  stiffness
    B += (1/2)*Qbar*(z_[i+1]**2-z_[i]**2)
    # bending or flexural laminate stiffness relating moments to curvatures
    D += (1/3)*Qbar*(z_[i+1]**3-z_[i]**3)  

A

B

D

A11_ = A[0,0]
B11_ = B[0,0]
D11_ = D[0,0]

# declare symbols for equation generation
#x,y,q = symbols('x,y,q')
th,x,y,z,q,a,b,C1,C2,C3,C4,C5,C6 = symbols('th,x,y,z,q,a,b,C1,C2,C3,C4,C5,C6')
strainx, strainy, strainxy, stressx, stress, stressxy = symbols('epsilonx,epsilony,gammaxy,sigmax,sigmay,sigmaxy')
ex,ey,exy,sx,sy,sxy = symbols('epsilon_x, epsilon_y, gamma_xy,sigma_x,sigma_y,tau_xy')

A11,A22,A66,A12,A16,A26,A66 = symbols('A11,A22,A66,A12,A16,A26,A66')
B11,B22,B66,B12,B16,B26,B66 = symbols('B11,B22,B66,B12,B16,B26,B66')
D11,D22,D66,D12,D16,D26,D66 = symbols('D11,D22,D66,D12,D16,D26,D66')
Nx,Ny,Nxy,Mx,My,Mxy = symbols('Nx,Ny,Nxy,Mx,My,Mxy')

##if use this, then reference  the function as u0(x), example  diff(u0(x),x,2)
#
#u0 = Function('u0')(x,y)
#v0 = Function('v0')(x,y)
#w0 = Function('w0')(x,y)

w0 = A11 / (A11*D11-B11**2) * ( q*x**4/24 - C2*x**3/6 - (C3- B11/A11*C1)*x**2/2 - C5*x - C6 )
w0

u0 = D11/(A11*D11 - B11**2 ) *C1*x + B11/(A11*D11-B11**2) * (q*x**3/6-C2*x**2/2-C3*x)+C4/A11
u0

# define boundary conditions

Nx = C1
Mx = -q*x**2/2+C2*x+C3

# simple support, pin pin
#bc1 = Mx.subs(x,+a/2)
#bc2 = Mx.subs(x,-a/2)
#bc3 = u0.subs(x,+a/2)
#bc4 = u0.subs(x,-a/2)
#bc5 = w0.subs(x,+a/2)
#bc6 = w0.subs(x,-a/2)

## pin-roller
#bc1 = Mx.subs(x,+a/2)
#bc2 = Mx.subs(x,-a/2)
#bc3 = Nx.subs(x,+a/2)
#bc4 = u0.subs(x,-a/2)
#bc5 = w0.subs(x,+a/2)
#bc6 = w0.subs(x,-a/2)

# fixed-pin
#bc1 = u0.subs(x,+a/2)
#bc2 = w0.subs(x,+a/2)
#bc3 = w0.diff(x).subs(x,+a/2)
#bc4 = u0.subs(x,-a/2)
#bc5 = Mx.subs(x,-a/2)
#bc6 = w0.subs(x,-a/2)

#fixed-pin
bc1 = u0.subs(x,+a/2)
bc2 = w0.subs(x,+a/2)
bc3 = Mx.subs(x,+a/2) #0
bc4 = w0.diff(x).subs(x,-a/2)
bc5 = u0.subs(x,-a/2) #0
bc6 = w0.subs(x,-a/2) #0

C = solve([bc1,bc2,bc3,bc4,bc5,bc6],[C1,C2,C3,C4,C5,C6])
C

C1_ = C[C1].subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_})
C2_ = C[C2].subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_})
C3_ = C[C3].subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_})
C4_ = C[C4].subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_})
C5_ = C[C5].subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_})
C6_ = C[C6].subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_})

u0 = u0.subs({C1:C[C1] , C2:C[C2], C3:C[C3], C4:C[C4], C5:C[C5], C6:C[C6]})
u0

w0 = w0.subs({C1:C[C1] , C2:C[C2], C3:C[C3], C4:C[C4], C5:C[C5], C6:C[C6]})
w0

# x displacement function u(x) 
u0f = u0.subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_,C1:C1_ , C2:C2_, C3:C3_, C4:C4_, C5:C5_, C6:C6_})
u0f

# z displacement function, w(x)
w0f = w0.subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_,C1:C1_ , C2:C2_, C3:C3_, C4:C4_, C5:C5_, C6:C6_})
w0f

# calculate strains based on the plate dispalcment

epsx = diff(u0,x) + 0.5* diff(w0,x)**2 - z*diff(w0,x,2)
epsx

epsy = 0.5* diff(w0,y)**2 - z*diff(w0,y,2)
epsy

epsxy = 0.5*(diff(u0,y) + diff(w0,x)*diff(w0,y)) - z*diff(w0,x,y) 
epsxy

epsx = epsx.subs({a:a_,b:b_,q:q_,A11:A11_, B11:B11_, D11:D11_,C1:C1_ , C2:C2_, C3:C3_, C4:C4_, C5:C5_, C6:C6_})
epsx

# Strain matrix in global coordinate system
epsbar = Matrix([[epsx],[epsy],[epsxy]])
epsbar

# plotting results 

# Sympy 3d plots
#from sympy.plotting import plot3d
#plot3d(w0f, (x,-a_/2,a_/2), (y,-b_/2,b_/2), title='beam deflection', xlabel='a,in', ylabel='b,in', zlabel='z,in');

# matplotlib plots
res = 250
X,Y = np.meshgrid(np.linspace(-a_/2,a_/2,res), np.linspace(-b_/2,b_/2,res))
w = sp.lambdify(x,w0f, "numpy")
fig = plt.figure('plate-warpage', figsize=(12, 8))

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, w(X), cmap=mpl.cm.jet, alpha=0.3) 
cset = ax.contourf(X, Y, w(X), cmap=mpl.cm.jet, alpha=0.3, zdir='z', offset=np.min(w(X))) 
#cbar = plt.colorbar(cset)

ax.set_xlabel('plate width,y-direction')
ax.set_ylabel('plate length,x-direction')
ax.set_zlabel('warpage')

#ax.view_init(elev=25, azim=-58)           # elevation and angle
#ax.dist=10          # distance

# plot contour lines
#CS = plt.contour(X, Y, w(X), cmap=mpl.cm.jet) ; cbar = plt.colorbar(CS) ; plt.clabel(CS, inline=1, fontsize=10)

plt.show()

for i,k in enumerate(range(0,2*nply,2)):

    Qbar = T1s(plyangle[i])**-1 @ Q @ T2s(plyangle[i]) 
    
    # stress is calcuated at top and bottom of each ply
    
    sigmabar = Qbar @ epsbar.subs({z:zmid_[i]})
    
    eps = T2s(plyangle[i]) @ epsbar.subs({z:zmid_[i]})
    
    sigma = Q @ eps
    
    for p in range(3):
        #plot3d(sigma[p], (x,-a_/2,a_/2), (y,-b_/2,b_/2), title='stress_%i at z=%f'%((p+1),zmid_[i]), xlabel='a,in', ylabel='b,in', zlabel='z,in')
        sigmaplot = sp.lambdify(x,sigma[p], "numpy")
        
        fig = plt.figure(i, figsize=(12, 8))
        
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, sigmaplot(X), cmap=mpl.cm.jet, alpha=0.3)    
        #ax.contourf(X, Y, sigmaplot(X), cmap=mpl.cm.jet, alpha=0.3, zdir='z', offset=np.min(w(X))) 
        
        plt.title('$\sigma_%i$, z=%f, $\Theta=%f$ ' % ( (p+1),zmid_[i], plyangle[i]*180/np.pi) )
        ax.set_xlabel('plate width,y-direction,in')
        ax.set_ylabel('plate length,x-direction, in')
        ax.set_zlabel('stress')

        plt.show()



