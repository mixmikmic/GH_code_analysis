#importing modules
import math
from __future__ import division

#Variable declaration
ni1=2.5*10**19;    #number of electron hole pairs
T1=300;     #temperature(K)
Eg1=0.72*1.6*10**-19;    #energy gap(J)
k=1.38*10**-23;     #boltzmann constant
T2=310;    #temperature(K)
Eg2=1.12*1.6*10**-19;    #energy gap(J)

#Calculation
x1=-Eg1/(2*k*T1);
y1=(T1**(3/2))*math.exp(x1);
x2=-Eg2/(2*k*T2);
y2=(T2**(3/2))*math.exp(x2);
ni=ni1*(y2/y1);          #number of electron hole pairs

#Result
print "number of electron hole pairs is",round(ni/10**16,2),"*10**16 per cubic metre"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
w=72.6;    #atomic weight
d=5400;    #density(kg/m**3)
Na=6.025*10**26;    #avagadro number
mew_e=0.4;    #mobility of electron(m**2/Vs)
mew_h=0.2;    #mobility of holes(m**2/Vs)
e=1.6*10**-19;
m=9.108*10**-31;    #mass(kg)
ni=2.1*10**19;      #number of electron hole pairs
Eg=0.7;    #band gap(eV)
k=1.38*10**-23;    #boltzmann constant
h=6.625*10**-34;    #plancks constant
T=300;     #temperature(K)

#Calculation
sigmab=ni*e*(mew_e+mew_h);    #intrinsic conductivity(ohm-1 m-1)
rhob=1/sigmab;     #resistivity(ohm m)
n=Na*d/w;     #number of germanium atoms per m**3
p=n/10**5;   #boron density
sigma=p*e*mew_h;
rho=1/sigma;

#Result
print "intrinsic conductivity is",round(sigma/10**4,3),"*10**4 ohm-1 m-1"
print "intrinsic resistivity is",round(rho*10**4,3),"*10**-4 ohm m"
print "answer varies due to rounding off errors"
print "number of germanium atoms per m**3 is",round(n/10**28,1),"*10**28"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;
RH=3.66*10**-4;    #hall coefficient(m**3/coulomb)
sigma=112;      #conductivity(ohm-1 m-1)

#Calculation
ne=3*math.pi/(8*RH*e);    #charge carrier density(per m**3)
mew_e=sigma/(e*ne);      #electron mobility(m**2/Vs)

#Result
print "charge carrier density is",int(ne/10**22),"*10**22 per m**3"
print "electron mobility is",round(mew_e,3),"m**2/Vs"

#importing modules
import math
from __future__ import division

#Variable declaration
mew_e=0.13;    #mobility of electron(m**2/Vs)
mew_h=0.05;    #mobility of holes(m**2/Vs)
e=1.6*10**-19;
ni=1.5*10**16;      #number of electron hole pairs
N=5*10**28;

#Calculation
sigma1=ni*e*(mew_e+mew_h);    #intrinsic conductivity(ohm-1 m-1)
ND=N/10**8;
n=ni**2/ND;
sigma2=ND*e*mew_e;     #conductivity(ohm-1 m-1)
sigma3=ND*e*mew_h;     #conductivity(ohm-1 m-1)

#Result
print "intrinsic conductivity is",round(sigma1*10**3,3),"*10**-3 ohm-1 m-1",sigma2
print "conductivity during donor impurity is",sigma2,"ohm-1 m-1"
print "conductivity during acceptor impurity is",int(sigma3),"ohm-1 m-1"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;
Eg=0.72;    #band gap(eV)
k=1.38*10**-23;    #boltzmann constant
T1=293;     #temperature(K)
T2=313;     #temperature(K)
sigma1=2;    #conductivity(mho m-1)

#Calculation
x=(Eg*e/(2*k))*((1/T1)-(1/T2));
y=round(x/2.303,3);
z=round(math.log10(sigma1),3);
log_sigma2=y+z;
sigma2=10**log_sigma2;     #conductivity(mho m-1)

#Result
print "conductivity is",round(sigma2,2),"mho m-1"

#importing modules
import math
from __future__ import division

#Variable declaration
ni=1.5*10**16
mu_n=1300*10**-4
mu_p=500*10**-4
e=1.6*10**-19
sigma=3*10**4

#Calculations
#Concentration in N-type
n1=sigma/(e*mu_n)
p1=ni**2/n1
#Concentration in P-type
p=sigma/(e*mu_p)
n2=(ni**2)/p

#Result
print"a)Concentration in N-type"
print"n =",round(n1*10**-24,3),"*10**24 m**-3"
print"Hence p =",round(p1/10**8,2),"*10**8 m**-3"
print"b)Concentration in P-type"
print"p =",round(p/10**24,2),"*10**24 m**-3"
print"Hence n =",round(n2/10**8,1),"*10**8 m**-3"

#importing modules
import math
from __future__ import division

#Variable declaration
i=10**-2
A=0.01*0.001
RH=3.66*10**-4
Bz=0.5

#Calculations
Jx=i/A
Ey=RH*(Bz*Jx)
Vy=Ey*0.01

#Result
print"Jx =",Jx,"ampere/m**2"
print"Ey =",round(Ey,3),"V/m"
print"Vy =",round(Vy*10**3,2),"mV"

#importing modules
import math
from __future__ import division

#Variable declaration
Ev=0
Ec=1.12
k=1.38*10**-23
T=300
mh=0.28
mc=0.12
e=1.6*10**-19
#Calculations
Ef=((Ec+Ev)/2)+((3*k*T)/(4*e))*math.log(mh/mc)

#Result
print"Position of fermi level =",round(Ef,4),"eV"

#importing modules
import math
from __future__ import division

#Variable declaration
ni=2.5*10**19
mu_e=0.38
mu_h=0.18
e=1.6*10**-19

#Calculations
sigmai=ni*e*(mu_e+mu_h)

#Result
print"Conductivity of intrinsic germanium at 300K =",round(sigmai,2),"ohm**-1 m**-1"

#importing modules
import math
from __future__ import division

#Variable declaration
m=9.1*10**-31
k=1.38*10**-23
T=300
h=6.626*10**-34
Eg=1.1
e=1.6*10**-19
mu_e=0.48
mu_h=0.013
#Calculations
ni=2*((2*math.pi*m*k*T)/h**2)**(3/2)*math.exp(-(Eg*e)/(2*k*T))
sigma=ni*e*(mu_e+mu_h)
                                                  
#Result
print"Conductivity =",round(sigma*10**3,4),"*10**-3 ohm**-1 m**-1"                                                  

#importing modules
import math
from __future__ import division

#Variable declaration
Na=5*10**23
Nd=3*10**23
ni=2*10**16
#Calculations
p=((Na-Nd)+(Na-Nd))/2

#Result
print"p =",p*10**-23,"*10**23 m**-3"
print"The electron concentration is given by n =",ni**2/p*10**-9,"*10**9 m**-3"

#importing modules
import math
from __future__ import division

#Variable declaration
Vh=37*10**-6
thick=1*10**-3
width=5
Iy=20*10**-3
Bz=0.5

#Calculations
Rh=(Vh*width*thick)/(width*Iy*Bz)

#Result
print"Rh =",round(Rh*10**6,1),"*10**-6 C**-1 m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
Vt=0.0258
mu_n=1300
mu_p=500

#Calculations
Dn=Vt*mu_n
Dp=Vt*mu_p

#Result
print"Dn =",Dn,"cm**2 s**-1"
print"Dp =",Dp,"cm**2 s**-1"

#importing modules
import math
from __future__ import division

#Variable declaration
ni=1.5*10**16
Nd=2*10**19
e=1.602*100**-19
mu_n=0.12

#Calculations
p=ni**2/Nd
E_c=e*Nd*mu_n

#Result
print"The hole concentration 'p' =",round(p*10**-13,3),"*10**13 /m**3"
print"'n'= Nd =",round(Nd*10**-19),"*10**19"
print"Electrical Conductivity =",round(E_c*10**19,3),"ohm**-1 m**-1"

#importing modules
import math
from __future__ import division

#Variable declaration
N=1/60
e=1.6*10**-19
ni=2.5*10**13
b=5*10**13
E=2

#Calculations
n=(b+math.sqrt(2*b**2))/2
mu_p=N/(3*e*ni)
mu_i=2*mu_p
np=ni**2
p=(ni**2)/n
e=1.6*10**-19
E=2
J=(e*E)*((n*mu_i)+(p*mu_p))
#Result
print"mu_p=",round(mu_p),"cm**2/V-s"
print"n=",round(n/10**13,4),"*10**13/cm**3"
print"p=",round(p*10**-13,4),"*10**13/cm**3"
print"J=",round(J*10**4,1),"A/m**2"
print"#Answer varies due to rounding of numbers"

#importing modules
import math
from __future__ import division

#Variable declaration
rho=47*10**-2
e=1.6*10**-19
mu_n=0.39
mu_p=0.19
E=10**4

#Calculations
ni=1/(rho*e*(mu_n+mu_p))
Dh=mu_p*E
De=mu_n*E

#Results
print"ni =",round(ni/10**19,3),"*10**19 /m**3"
print"Drift velocity of holes",Dh,"ms**-1"
print"Drift velocity of electrons=",De,"ms**-1"

