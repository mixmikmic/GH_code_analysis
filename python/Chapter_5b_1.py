#importing modules
import math
from __future__ import division

#Variable declaration
El=10**-2*50;       #energy loss(J)
H=El*60;      #heat produced(J)
d=7.7*10**3;    #iron rod(kg/m**3)
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
B=mew0*i/(2*R);    #magnetic field at the centre(weber/m**2)
A=math.pi*R**2;
d=i*A;       #dipole moment(ampere/m**2)

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

