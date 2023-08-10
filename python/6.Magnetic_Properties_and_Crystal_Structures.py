#importing modules
import math
from __future__ import division

#Variable declaration
El=10**-2*50;      #energy loss(J)
H=El*60;           #heat produced(J)
d=7.7*10**3;       #iron rod(kg/m**3)
s=0.462*10**-3;    #specific heat(J/kg K)

#Calculation
theta=H/(d*s);     #temperature rise(K)

#Result
print "temperature rise is",round(theta,2),"K"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;    #charge(coulomb)
new=6.8*10**15;   #frequency(revolutions per second)
mew0=4*math.pi*10**-7;
R=5.1*10**-11;     #radius(m)

#Calculation
i=round(e*new,4);          #current(ampere)
B=mew0*i/(2*R);            #magnetic field at the centre(weber/m**2)
A=math.pi*R**2;
d=i*A;                    #dipole moment(ampere/m**2)

#Result
print "magnetic field at the centre is",round(B),"weber/m**2"
print "dipole moment is",round(d*10**24),"*10**-24 ampere/m**2"

#importing modules
import math
from __future__ import division

#Variable declaration
chi=0.5*10**-5;    #magnetic susceptibility
H=10**6;     #field strength(ampere/m)
mew0=4*math.pi*10**-7;

#Calculation
I=chi*H;     #intensity of magnetisation(ampere/m)
B=mew0*(I+H);    #flux density in material(weber/m**2)

#Result
print "intensity of magnetisation is",I,"ampere/m"
print "flux density in material is",round(B,3),"weber/m**2"

#importing modules
import math
from __future__ import division

#Variable declaration
B=9.27*10**-24;      #bohr magneton(ampere m**2)
a=2.86*10**-10;      #edge(m)
Is=1.76*10**6;       #saturation value of magnetisation(ampere/m)

#Calculation
N=2/a**3;
mew_bar=Is/N;      #number of Bohr magnetons(ampere m**2)
mew_bar=mew_bar/B;      #number of Bohr magnetons(bohr magneon/atom)

#Result
print "number of Bohr magnetons is",round(mew_bar,2),"bohr magneon/atom"

#importing modules
import math
from __future__ import division

#Variable declaration
mew0=4*math.pi*10**-7;
H=9.27*10**-24;      #bohr magneton(ampere m**2)
beta=10**6;      #field(ampere/m)
k=1.38*10**-23;    #boltzmann constant
T=303;    #temperature(K)

#Calculation
mm=mew0*H*beta/(k*T);    #average magnetic moment(bohr magneton/spin)

#Result
print "average magnetic moment is",round(mm*10**3,2),"*10**-3 bohr magneton/spin"

#importing modules
import math
from __future__ import division

#Variable declaration
A=94;      #area(m**2)
vy=0.1;    #value of length(weber/m**2)
vx=20;     #value of unit length
n=50;      #number of magnetization cycles
d=7650;    #density(kg/m**3)

#Calculation
h=A*vy*vx;     #hysteresis loss per cycle(J/m**3)
hs=h*n;       #hysteresis loss per second(watt/m**3)
pl=hs/d;      #power loss(watt/kg)

#Result
print "hysteresis loss per cycle is",h,"J/m**3"
print "hysteresis loss per second is",hs,"watt/m**3"
print "power loss is",round(pl,2),"watt/kg"

import math
from __future__ import division

#variable declaration
d=2.351                 #bond lenght
N=6.02*10**26           #Avagadro number
n=8                     #number of atoms in unit cell
A=28.09                 #Atomin mass of silicon
m=6.02*10**26           #1mole

#Calculations
a=(4*d)/math.sqrt(3)
p=(n*A)/((a*10**-10)*m)    #density

#Result
print "a=",round(a,2),"Angstorm"
print "density =",round(p*10**16,2),"kg/m**3"
print"#Answer given in the textbook is wrong"

import math
from __future__ import division
from sympy import Symbol

#Variable declaration
r=Symbol('r')

#Calculation
a1=4*r/math.sqrt(3);
R1=(a1/2)-r;           #radius of largest sphere
a2=4*r/math.sqrt(2);
R2=(a2/2)-r;       #maximum radius of sphere

#Result
print "radius of largest sphere is",R1
print "maximum radius of sphere is",R2    

import math
from __future__ import division

#variable declaration
r1=1.258            #Atomic radius of BCC
r2=1.292            #Atomic radius of FCC

#calculations
a1=(4*r1)/math.sqrt(3)       #in BCC
b1=((a1)**3)*10**-30         #Unit cell volume
v1=(b1)/2                    #Volume occupied by one atom
a2=2*math.sqrt(2)*r2         #in FCC
b2=(a2)**3*10**-30                   #Unit cell volume
v2=(b2)/4                    #Volume occupied by one atom  
v_c=((v1)-(v2))*100/(v1)     #Volume Change in % 
d_c=((v1)-(v2))*100/(v2)     #Density Change in %

#Results
print "a1=",round(a1,3),"Angstrom" 
print "Unit cell volume =a1**3 =",round((b1)/10**-30,3),"*10**-30 m**3"
print "Volume occupied by one atom =",round(v1/10**-30,2),"*10**-30 m**3"
print "a2=",round(a2,3),"Angstorm"
print "Unit cell volume =a2**3 =",round((b2)/10**-30,3),"*10**-30 m**3"
print "Volume occupied by one atom =",round(v2/10**-30,2),"*10**-30 m**3"
print "Volume Change in % =",round(v_c,3)
print "Density Change in % =",round(d_c,2)
print "Thus the increase of density or the decrease of volume is about 0.5%"

import math
from __future__ import division

#variable declaration
n=4     
M=58.5                  #Molecular wt. of NaCl
N=6.02*10**26           #Avagadro number
rho=2180                #density

#Calculations
a=((n*M)/(N*rho))**(1/3)    
s=a/2

#Result
print "a=",round(a/10**-9,3),"*10**-9 metre"
print "spacing between the nearest neighbouring ions =",round(s/10**-9,4),"nm"

import math
from __future__ import division

#variable declaration
n=4     
A=63.55                 #Atomic wt. of NaCl
N=6.02*10**26           #Avagadro number
rho=8930                #density

#Calculations
a=((n*A)/(N*rho))**(1/3)      #Lattice Constant

#Result
print "lattice constant, a=",round(a*10**9,2),"nm"

import math

#variable declaration
r=0.123                  #Atomic radius
n=4
A=55.8                   #Atomic wt
a=2*math.sqrt(2) 
N=6.02*10**26           #Avagadro number

#Calculations
rho=(n*A)/((a*r*10**-9)**3*N)

#Result
print "Density of iron =",round(rho),"kg/m**-3"

