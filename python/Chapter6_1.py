#importing modules
import math
from __future__ import division

#Variable declaration
rho=1.54*10**-8;    #resistivity(ohm m)
n=5.8*10**28;       #conduction electrons(per m**3)
e=1.6*10**-19;      #charge(c)
m=9.1*10**-31;      #mass(kg)

#Calculation
towr=m/(n*e**2*rho);     #relaxation time(sec)

#Result
print "relaxation time is",round(towr*10**14,4),"*10**-14 sec"

#importing modules
import math
from __future__ import division

#Variable declaration
T=300;    #temperature(K)
n=8.5*10**28;    #density(per m**3)
rho=1.69*10**-8;   #resistivity(ohm/m**3)
e=1.6*10**-19;      #charge(c)
m=9.11*10**-31;      #mass(kg)
Kb=1.38*10**-23;     #boltzmann constant(J/k)

#Calculation
rho=math.sqrt(3*Kb*m*T)/(n*e**2*rho);     #mean free path(m)

#Result
print "mean free path is",round(rho*10**9,2),"*10**-9 m"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
rho=1.43*10**-8;    #resistivity(ohm m)
n=6.5*10**28;       #conduction electrons(per m**3)
e=1.6*10**-19;      #charge(c)
m=9.1*10**-34;      #mass(kg)

#Calculation
towr=m/(n*e**2*rho);     #relaxation time(sec)

#Result
print "relaxation time is",round(towr*10**17,3),"*10**-17 sec"
print "answer in the book varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
PE=1/100;     #probability
E_EF=0.5;     #energy difference

#Calculation
x=math.log((1/PE)-1);
T=E_EF/x;     #temperature(K)

#Result
print "temperature is",round(T,4),"K"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
d=8.92*10**3;     #density(kg/m**3)
rho=1.73*10**-8;    #resistivity(ohm m)
M=63.5;    #atomic weight
N=6.02*10**26;    #avagadro number
e=1.6*10**-19;      #charge(c)
m=9.1*10**-31;      #mass(kg)

#Calculation
n=d*N/M;
mew=1/(rho*n*e);      #mobility(m/Vs)
tow=m/(n*e**2*rho);   #average time(sec)

#Result
print "mobility is",round(mew*10**2,3),"*10**-2 m/Vs"
print "average time is",round(tow*10**14,2),"*10**-14 sec"

#importing modules
import math
from __future__ import division

#Variable declaration
EF=5.5;     #energy(eV)
FE=10/100;   #probability
e=1.6*10**-19;      #charge(c)
Kb=1.38*10**-23;     #boltzmann constant(J/k)

#Calculation
E=EF+(EF/100);    
x=(E-EF)*e;
y=x/Kb;
z=(1/FE)-1;
T=y/math.log(z);      #temperature(K)

#Result
print "temperature is",round(T,1),"K"

#importing modules
import math
from __future__ import division

#Variable declaration
Kb=1.38*10**-23;     #boltzmann constant(J/k)
T=303;     #temperature(K)
e=1.6*10**-19;      #charge(c)
MH=2*1.008*1.67*10**-27;   #mass(kg)   

#Calculation
KE=3*Kb*T/(2*e);     #kinetic energy(eV)
cbar=math.sqrt(3*Kb*T/MH);     #velocity(m/s)

#Result
print "kinetic energy is",round(KE*10**3,1),"*10**-3 eV"
print "velocity is",round(cbar,2),"m/s"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
rho=10**4;      #density of silver(kg/m**3)
N=6.02*10**26;    #avagadro number
e=1.6*10**-19;      #charge(c)
m=9.1*10**-31;      #mass(kg)
MA=107.9;     #atomic weight(kg)
sigma=7*10**7;    #conductivity(per ohm m)

#Calculation
n=rho*N/MA;       #density of electrons(per m**3)
mew=sigma/(n*e*10**2);    #mobility of electrons(m**2/Vs)
tow=sigma*m*10**15/(n*e**2);    #collision time(n sec)

#Result
print "density of electrons is",round(n/10**26,1),"*10**26 per m**3"
print "mobility of electrons is",round(mew*10**5,4),"*10**-5 m**2/Vs"
print "collision time is",round(tow,1),"n sec"

#importing modules
import math
from __future__ import division

#Variable declaration
Ee=10;     #electron kinetic energy(eV)
Ep=10;     #proton kinetic energy(eV)
e=1.6*10**-19;      #charge(c)
me=9.1*10**-31;      #mass(kg)
mp=1.67*10**-27;     #mass(kg)

#Calculation
cebar=math.sqrt(2*Ee*e/me);    #electron velocity(m/s)
cpbar=math.sqrt(2*Ep*e/mp);    #proton velocity(m/s)

#Result
print "electron velocity is",round(cebar/10**6,3),"*10**6 m/s"
print "proton velocity is",round(cpbar/10**3,3),"*10**3 m/s"
print "answers given in the book are wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
A=10*10**-6;     #area(m**2)
i=100;    #current(amp)
n=8.5*10**28;    #number of electrons
e=1.6*10**-19;      #charge(c)

#Calculation
vd=i/(n*A*e);     #drift velocity(m/s)

#Result
print "drift velocity is",round(vd*10**4,5),"*10**-4 m/s"

#importing modules
import math
from __future__ import division

#Variable declaration
Kb=1.38*10**-23;     #boltzmann constant(J/k)
m=9.1*10**-31;      #mass(kg)
tow=3*10**-14;    #relaxation time(sec)
n=8*10**28;      #density of electrons(per m**3)
T=273;     #temperature(K)

#Calculation
sigma_T=3*n*tow*T*Kb**2/(2*m);      #thermal conductivity(W/mK)

#Result
print "thermal conductivity is",round(sigma_T,3),"W/mK"

