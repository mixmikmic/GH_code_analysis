import math
from __future__ import division

#variable declaration
r1 = 2;               #in radians
r2 = 3;              #in radians
d1 = 4;               #Converting from mm to radians
d2 = 6;              #Converting from mm to radians

#calculations
D = (r2-r1)/(d2*10**3-d1*10**3)

#Result
print "Divergence =",round(D*10**3,3),"*10**-3 radian"

import math
from __future__ import division
from sympy import *
#variable declaration
C=3*10**8                 #The speed of light
L=6943                    #Wavelength
T=300                        #Temperature in Kelvin
h=6.626*10**-34              #Planck constant 
k=1.38*10**-23               #Boltzmann's constant

#Calculations

V=(C)/(L*10**-10)
R=math.exp(h*V/(k*T))

#Result
print "Frequency (V) =",round(V/10**14,2),"*10**14 Hz"
print "Relative Population=",round(R/10**30,3),"*10**30"

import math

C=3*10**8                     #Velocity of light
W=632.8*10**-9                #wavelength
P=2.3
t=1
h=6.626*10**-34              #Planck constant 
S=1*10**-6

#Calculations
V=C/W                        #Frequency
n=((P*10**-3)*t)/(h*V)       #no.of photons emitted
PD=P*10**-3/S

#Result
print "Frequency=",round(V/10**14,2),"*10**14 Hz"
print "no.of photons emitted=",round(n/10**15,3),"*10**15 photons/sec"
print "Power density =",round(PD/1000,1),"kWm**-2"

import math

#variable declaration
h=6.626*10**-34              #Planck constant 
C=3*10**8                    #Velocity of light
E_g=1.44                     #bandgap 

#calculations
W=(h*C)*10**10/(E_g*1.6*10**-19)

#Result
print "Wavelenght =",round(W),"Angstrom"

import math

#variable declaration
W=1.55                       #wavelength

#Calculations
E_g=(1.24)/W                 #Bandgap in eV           

#Result
print "Band gap =",E_g,"eV"

