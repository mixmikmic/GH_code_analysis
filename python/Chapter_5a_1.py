#importing modules
import math
from __future__ import division

#Variable declaration
rho=5*10**16;   #resistivity(ohm m)
l=5*10**-2;    #thickness(m)
b=8*10**-2;    #length(m)
w=3*10**-2;    #width(m)

#Calculation
A=b*w;    #area(m**2)
Rv=rho*l/A;    
X=l+b;      #length(m)
Y=w;      #perpendicular(m)
Rs=Rv*X/Y;    
Ri=Rs*Rv/(Rs+Rv);       #insulation resistance(ohm)

#Result
print "insulation resistance is",round(Ri/10**18,2),"*10**18 ohm"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
epsilon0=8.84*10**-12;
R=0.55*10**-10;    #radius(m)
N=2.7*10**25;      #number of atoms

#Calculation
alpha_e=4*math.pi*epsilon0*R**3;    #polarisability of He(farad m**2)
epsilonr=1+(N*alpha_e/epsilon0);      #relative permittivity

#Result
print "polarisability of He is",round(alpha_e*10**40,3),"*10**-40 farad m**2"
print "relative permittivity is",round(epsilonr,6)
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
A=360*10**-4;    #area(m**2)
V=15;    #voltage(V)
C=6*10**-6;     #capacitance(farad)
epsilonr=8;
epsilon0=8.84*10**-12;

#Calculation
E=V*C/(epsilon0*epsilonr*A);     #field strength(V/m)
dm=epsilon0*(epsilonr-1)*V*A;    #total dipole moment(Cm)

#Result
print "field strength is",round(E/10**7,3),"*10**7 V/m"
print "total dipole moment is",round(dm*10**12,1),"*10**-12 Cm"

#importing modules
import math
from __future__ import division

#Variable declaration
epsilonr=4.36;      #dielectric constant
t=2.8*10**-2;       #loss tangent(t)
N=4*10**28;         #number of electrons
epsilon0=8.84*10**-12;      

#Calculation
epsilon_r = epsilonr*t;
epsilonstar = (complex(epsilonr,-epsilon_r));
alphastar = (epsilonstar-1)/(epsilonstar+2);
alpha_star = 3*epsilon0*alphastar/N;             #complex polarizability(Fm**2)

#Result
print "the complex polarizability is",alpha_star*10**40,"*10**-40 F-m**2"
print "answer cant be rouned off to 2 decimals as given in the textbook. Since it is a complex number and complex cant be converted to float"

