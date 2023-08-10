import math
from __future__ import division

#variable declaration
n1=1.50          #Core refractive index
n2=1.47          #Cladding refractive index

#Calculations
C_a=math.asin(n2/n1)        #Critical angle       
N_a=(n1**2-n2**2)**(1/2)
A_a=math.asin(N_a)

#Results
print "The Critical angle =",round(C_a*180/math.pi,1),"degrees"
print "The numerical aperture =",round(N_a,2)
print "The acceptance angle =",round(A_a*180/math.pi,1),"degrees"

import math
from __future__ import division

#variable declaration
d=50                #diameter
N_a=0.2             #Numerical aperture
lamda=1             #wavelength

#Calculations
N=4.9*(((d*10**-6*N_a)/(lamda*10**-6))**2)

#Result
print "N =",N
print "Fiber can support",N,"guided modes"
print "In graded index fiber, No.of modes propogated inside the fiber =",N/2,"only"

import math
from __future__ import division

#variable declaration
d=50                #diameter
n1=1.450
n2=1.447
lamda=1             #wavelength

#Calculations
N_a=(n1**2-n2**2)   #Numerical aperture
N=4.9*(((d*10**-6*N_a)/(lamda*10**-6))**2)

#Results
print "Numerical aperture =",N_a
print "No. of modes that can be propogated =",round(N)

import math
from __future__ import division

#variable declaration
delta=0.05          
n1=1.46

#Calculation
N_a=n1*(2*delta)**(1/2)     #Numerical aperture

#Result
print "Numerical aperture =",round(N_a,2)

import math
from __future__ import division

#variable declaration
a=50
n1=1.53
n2=1.50
lamda=1             #wavelength

#Calculations
N_a=(n1**2-n2**2)   #Numerical aperture
V=((2*math.pi*a)/lamda)*N_a**(1/2)

#Result
print "V number =",round(V,2)
print "maximum no.of modes propogating through fiber =",round(N)

import math
from __future__ import division

#variable declaration
a=100
N_a=0.3               #Numerical aperture
lamda=850             #wavelength

#Calculations
V_n=(2*(math.pi)**2*a**2*10**-12*N_a**2)/lamda**2*10**-18
#Result
print "Number of modes =",round(V_n/10**-36),"modes"
print "No.of modes is doubled to account for the two possible polarisations"
print "Total No.of modes =",round(V_n/10**-36)*2

import math

#variable declaration
a=5;
n1=1.48;
delta=0.01;
V=25;

#Calculation
lamda=(math.pi*(a*10**-6)*n1*math.sqrt(2*delta))/V   # Cutoff Wavelength

#Result
print "Cutoff Wavellength =",round(lamda*10**7,3),"micro m."

import math

#variable declaration
V=2.405
lamda=1.3
N_a=0.05

#Calculations
a_max=(V*lamda)/(2*math.pi*N_a)

#Result
print "Maximum core radius=",round(a_max,2),"micro m"

import math
from __future__ import division

#variable declaration
N_a=0.3
gamma=45

#Calculations
theta_a=math.asin(N_a)
theta_as=math.asin((N_a)/math.cos(gamma))

#Results
print "Acceptance angle, theta_a =",round(theta_a*180/math.pi,2),"degrees"
print "For skew rays,theta_as ",round(theta_as*180/math.pi,2),"degrees"
print"#Answer given in the textbook is wrong"

import math
from __future__ import division

#variable declaration
n1=1.53
delta=0.0196

#Calculations
N_a=n1*(2*delta)**(1/2)
A_a=math.asin(N_a)
#Result
print "Numerical aperture =",round(N_a,3)
print "Acceptance angle =",round(A_a*180/math.pi,2),"degrees"

import math
from __future__ import division

#variable declaration
n1=1.480
n2=1.465
V=2.405
lamda=850*10**-9

#Calculations
delta=(n1**2-n2**2)/(2*n1**2)
a=(V*lamda*10**-9)/(2*math.pi*n1*math.sqrt(2*delta))

#Results
print "delta =",round(delta,2)
print "Core radius,a =",round(a*10**15,2),"micro m"

import math
from __future__ import division

#variable declaration
n1=1.5
n2=1.49
a=25

#Calculations
C_a=math.asin(n2/n1)           #Critical angle
L=2*a*math.tan(C_a)             
N_r=10**6/L                    

#Result
print "Critical angle=",round(C_a*180/math.pi,2),"degrees"
print "Fiber length covered in one reflection=",round(L,2),"micro m"
print "Total no.of reflections per metre=",round(N_r)
print "Since L=1m, Total dist. travelled by light over one metre of fiber =",round(1/math.sin(C_a),4),"m"

import math
from __future__ import division

#variable declaration
alpha=1.85
lamda=1.3*10**-6
a=25*10**-6
N_a=0.21

#Calculations
V_n=((2*math.pi**2)*a**2*N_a**2)/lamda**2
N_m=(alpha/(alpha+2))*V_n

print "No.of modes =",round(N_m,2),"=155(approx)"
print "Taking the two possible polarizations, Total No.of nodes =",round(N_m*2)

import math
from __future__ import division

#variable declaration
P_i=100
P_o=2
L=10

#Calculations
S=(10/L)*math.log(P_i/P_o)
O=S*L

#Result
print "a)Signal attention per unit length =",round(S,1),"dB km**-1"
print "b)Overall signal attenuation =",round(O),"dB"
print "#Answer given in the textbook is wrong"

import math
from __future__ import division

#variable declaration
L=10
n1=1.55
delta=0.026
C=3*10**5

#Calculations
delta_T=(L*n1*delta)/C
B_W=10/(2*delta_T)

#Result
print "Total dispersion =",round(delta_T/10**-9,1),"ns"
print "Bandwidth length product =",round(B_W/10**5,2),"Hz-km"
print "#Answer given in the text book is wrong"

