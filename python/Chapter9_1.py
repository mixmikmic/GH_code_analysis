#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;     #charge(c)
ni=2.4*10**19;    #particle density(per m**3)
mew_e=0.39;        #electron mobility(m**2/Vs)
mew_h=0.19;       #hole mobility(m**2/Vs)

#Calculation
rho=1/(ni*e*(mew_e+mew_h));       #resistivity(ohm m)

#Result
print "resistivity is",round(rho,5),"ohm m"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;     #charge(c)
ni=1.5*10**16;    #particle density(per m**3)
mew_e=0.13;        #electron mobility(m**2/Vs)
mew_h=0.048;       #hole mobility(m**2/Vs)
ND=10**23;     #density(per m**3)

#Calculation
sigma_i=ni*e*(mew_e+mew_h);          #conductivity(s)
sigma=ND*mew_e*e;       #conductivity(s)
P=ni**2/ND;            #equilibrium hole concentration(per m**3)

#Result
print "conductivity is",round(sigma_i*10**3,2),"*10**-3 s"
print "conductivity is",sigma/10**3,"*10**3 s"
print "answer in the book varies due to rounding off errors"
print "equilibrium hole concentration is",P/10**9,"*10**9 per m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;     #charge(c)
ni=1.5*10**16;    #particle density(per m**3)
mew_e=0.13;        #electron mobility(m**2/Vs)
mew_h=0.05;       #hole mobility(m**2/Vs)
ND=5*10**20;     #density(per m**3)

#Calculation
sigma=ni*e*(mew_e+mew_h);          #intrinsic conductivity(s)
sigma_d=ND*e*mew_e;       #conductivity during donor impurity(ohm-1 m-1)
sigma_a=ND*e*mew_h;       #conductivity during acceptor impurity(ohm-1 m-1)

#Result
print "intrinsic conductivity is",sigma*10**3,"*10**-3 ohm-1 m-1"
print "conductivity during donor impurity is",sigma_d,"ohm-1 m-1"
print "conductivity during donor impurity is",sigma_a,"ohm-1 m-1"

#importing modules
import math
from __future__ import division

#Variable declaration
RH=3.66*10**-4;     #hall coefficient(m**3/c)
rho=8.93*10**-3;    #resistivity(m)
e=1.6*10**-19;     #charge(c)

#Calculation
mew=RH/rho;        #mobility(m**2/Vs)
n=1/(RH*e);      #density of atoms(per m**3)

#Result
print "mobility is",round(mew,5),"m**2/Vs"
print "answer in the book varies due to rounding off errors"
print "density of atoms is",round(n/10**22,1),"*10**22 per m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
w=72.6;       #atomic weight
e=1.6*10**-19;     #charge(c)
mew_e=0.4;        #electron mobility(m**2/Vs)
mew_h=0.2;       #hole mobility(m**2/Vs)
T=300;         #temperature(K)
x=4.83*10**21;
Eg=0.7;      #band gap(eV)
y=0.052;

#Calculation
ni=x*(T**(3/2))*math.exp(-Eg/y);     #carrier density(per m**3)
sigma=ni*e*(mew_e+mew_h);         #conductivity(ohm-1 m-1)

#Result
print "carrier density is",round(ni/10**19,2),"*10**19 per m**3"
print "conductivity is",round(sigma,2),"ohm-1 m-1"
print "answer in the book varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
T1=293;     #temperature(K)
T2=305;     #temperature(K)
sigma1=2;    
sigma2=4.5; 
KB=1.38*10**-23;   #boltzmann constant

#Calculation
x=((1/T1)-(1/T2));
y=math.log(sigma2/sigma1);
z=3*math.log(T2/T1)/2;
Eg=2*KB*(y+z)/(e*x);    #energy band gap(eV)

#Result
print "energy band gap is",round(Eg,2),"eV"
print "answer in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;     #charge(c)
mew_e=0.19;        #electron mobility(m**2/Vs)
T=300;         #temperature(K)
KB=1.38*10**-23;   #boltzmann constant

#Calculation
Dn=mew_e*KB*T/e;    #diffusion coefficient(m**2/sec)

#Result
print "diffusion coefficient is",round(Dn*10**3,1),"*10**-3 m**2/sec"

#importing modules
import math
from __future__ import division

#Variable declaration
sigma=2.12;     #conductivity(ohm-1 m-1)
T=300;      #temperature(K)
e=1.6*10**-19;     #charge(c)
mew_e=0.36;        #electron mobility(m**2/Vs)
mew_h=0.7;       #hole mobility(m**2/Vs)
C=4.83*10**21;
KB=1.38*10**-23;   #boltzmann constant

#Calculation
ni=sigma/(e*(mew_e+mew_h));     #carrier density(per m**3)
x=C*T**(3/2)/ni;
Eg=2*KB*T*math.log(x)/e;      #energy gap(eV)

#Result
print "carrier density is",ni,"per m**3"
print "energy gap is",round(Eg,2),"eV"
print "answer in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=6.408*10**-20;    #energy gap of semiconductor(J)
T1=273;    #temperature(K)
T2=323;    #temperature(K)
T3=373;    #temperature(K)
KB=1.38*10**-23;   #boltzmann constant

#Calculation
FE1=1/(1+math.exp(Eg/(2*KB*T1)));     #probability of occupation at 0C(eV)
FE2=1/(1+math.exp(Eg/(2*KB*T2)));     #probability of occupation at 50C(eV)
FE3=1/(1+math.exp(Eg/(2*KB*T3)));     #probability of occupation at 100C(eV)

#Result
print "probability of occupation at 0C is",round(FE1*10**4,3),"*10**-4 eV"
print "probability of occupation at 50C is",round(FE2*10**4,2),"*10**-4 eV"
print "probability of occupation at 100C is",round(FE3*10**4,2),"*10**-4 eV"

#importing modules
import math
from __future__ import division

#Variable declaration
Eg=1.9224*10**-19;    #energy gap of semiconductor(J)
T1=600;    #temperature(K)
T2=300;    #temperature(K)
x=-1.666*10**-3;
KB=1.38*10**-23;   #boltzmann constant

#Calculation
T=(1/T1)-(1/T2);
r=math.exp(x*(-Eg/(2*KB)));     #ratio between conductivity

#Result
print "ratio between conductivity is",round(r/10**5,3),"*10**5"

#importing modules
import math
from __future__ import division

#Variable declaration
ni=2.5*10**19;      #charge carriers(per m**3)
r=10**-6;    #ratio
e=1.6*10**-19;     #charge(c)
mew_e=0.36;        #electron mobility(m**2/Vs)
mew_h=0.18;       #hole mobility(m**2/Vs)
N=4.2*10**28;     #number of atoms(per m**3)

#Calculation
Ne=r*N;       #number of impurity atoms(per m**3)
Nh=ni**2/Ne;    
sigma=(Ne*e*mew_e)+(Nh*e*mew_h);      #conductivity(ohm m)
rho=1/sigma;     #resistivity of material(per ohm m)

#Result
print "resistivity of material is",round(rho*10**4,4),"*10**-4 ohm m"

#importing modules
import math
from __future__ import division

#Variable declaration
n=5*10**17;     #concentration(m**3)
vd=350;    #drift velocity(m/s)
E=1000;    #electric field(V/m)
e=1.6*10**-19;     #charge(c)

#Calculation
sigma=n*e*vd/E;    #conductivity(per ohm m)

#Result
print "conductivity is",sigma,"per ohm m"

#importing modules
import math
from __future__ import division

#Variable declaration
sigmae=2.2*10**-4;       #conductivity(ohm/m)
mew_e=125*10**-3;       #electron mobility(m**2/Vs)
e=1.602*10**-19;     #charge(c)

#Calculation
ne=sigmae/(e*mew_e);     #concentration(per m**3)

#Result
print "concentration is",round(ne/10**16,1),"*10**16 per m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
RH=3.66*10**-4;      #hall coefficient(m*3/c)
rho_i=8.93*10**-3;    #resistivity(ohm m)
e=1.602*10**-19;     #charge(c)

#Calculation
nh=1/(RH*e);      #density of charge carriers(per m**3)
mewh=1/(rho_i*nh*e);     #mobility of charge carriers(m**2/Vs)

#Result
print "density of charge carriers is",round(nh/10**22,4),"*10**22 per m**3"
print "mobility of charge carriers is",round(mewh,3),"m**2/Vs"

#importing modules
import math
from __future__ import division

#Variable declaration
I=3*10**-3;    #current(A)
RH=3.66*10**-4;    #hall coefficient(m**3/C)
e=1.6*10**-19;     #charge(c)
d=2*10**-2;
z=1*10**-3;
B=1;        #magnetic field(wb/m**2)

#Calculation
w=d*z;     #width(m**2)
A=w;     #area(m**2)
EH=RH*I*B/A;    
VH=EH*d*10**3;     #hall voltage(mV)
n=1/(RH*e);     #charge carrier concentration(per m**3)

#Result
print "hall voltage is",round(VH,1),"mV"
print "charge carrier concentration is",round(n/10**22,2),"*10**22 per m**3"

