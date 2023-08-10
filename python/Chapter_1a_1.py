#importing modules
import math
from __future__ import division
from sympy import Symbol
from sympy import diff
import numpy as np

#Variable declaration
n=1;
m=9;
a=Symbol('a')
b=Symbol('b')
r=Symbol('r')

#Calculation
y=(-a/(r**n))+(b/(r**m));
y=diff(y,r);
y=diff(y,r);

#Result
print y

import math
from __future__ import division

#Variable declaration
a=7.68*10**-29;     
r0=2.5*10**-10;    #radius(m)

#Calculation
b=a*(r0**8)/9;
y=((-2*a*r0**8)+(90*b))/r0**11;    
E=y/r0;           #young's modulus(Pa)

#Result
print "young's modulus is",int(E/10**9),"GPa"

import math

#variable declarations
d=((1.98)*10**-29)*1/3;        #dipole moment
b=(0.92);                      #bond length
EC=d/(b*10**-10);              #Effective charge

#Result
print "Effective charge =",round((EC*10**19),2),"*10**-29 coulomb"

import math
from __future__ import division

#variable declaration
A=1.748                 #Madelung Constant    
N=6.02*10**26           #Avagadro Number
e=1.6*10**-19
n=9.5
r=(0.324*10**-9)*10**3
E=8.85*10**-12
#Calculations
U=((N*A*(e)**2)/(4*math.pi*E*r))*(1-1/n)       #Cohesive energy

#Result
print "Cohesive energy =",round(U/10**3,1),"*10**3 kJ/kmol"
print "#Answer varies due to rounding of numbers"

import math
from __future__ import division
#variable declaration
I=5;                       #Ionisation energy
A=4;                       #Electron Affinity
e=(1.6*10**-19)
E=8.85*10**-12            #epsilon constant
r=0.5*10**-19             #dist between A and B

#Calculations
C=-(e**2/(4*math.pi*E*r*e))/10**10    #Coulomb energy
E_c=I-A+C                             #Energy required

#Result
print "Coulomb energy =",round(C,2),"eV"
print "Energy required =",round(E_c,2),"eV"

import math
from __future__ import division

#variable declaration
I=5.14;                    #Ionization energy
A=3.65;                    #Electron Affinity
e=(1.6*10**-19);
E=8.85*10**-12; 
#calculations
E_c=I-A                        #Energy required
r=e**2/(4*math.pi*E*E_c*e)     #Distance of separation

#Result
print "Energy required=",E_c,"eV"
print "Distance of separation =",round(r/10**-10,2),"Angstrom"

import math
from __future__ import division

#variable declaration 
I=5.14;                    #Ionization energy
A=3.65;                    #Electron Affinity
e=(1.6*10**-19);
E=8.85*10**-12; 
r=236*10**-12;

#Calculations
E_c=I-A                        #Energy required
C=-(e**2/(4*math.pi*E*r*e))    #Potentential energy in eV
BE=-(E_c+C)                    #Bond Energy
#Result
print "Energy required=",E_c,"eV"
print "Energy required =",round(C,1),"eV"
print "Bond Energy =",round(BE,2),"eV"

