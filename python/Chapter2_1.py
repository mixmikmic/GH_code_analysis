#importing modules
import math
from __future__ import division

#Variable declaration
N=6.02*10**26;           #Avagadro Number
n=8;    #number of atoms
a=5.6*10**-10;    #lattice constant(m)
M=72.59;     #atomic weight(amu)

#Calculation
rho=n*M/(a**3*N);     #density(kg/m**3)

#Result
print "density is",round(rho,3),"kg/m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
N=6.02*10**23;           #Avagadro Number
n=2;
rho=7860;    #density(kg/m**3)
M=55.85;    #atomic weight(amu)

#Calculation
a=(n*M/(rho*N))**(1/3)*10**8;    #lattice constant(angstrom)

#Result
print "lattice constant is",round(a,4),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
N=6.02*10**26;           #Avagadro Number
n=2;
rho=530;    #density(kg/m**3)
M=6.94;    #atomic weight(amu)

#Calculation
a=(n*M/(rho*N))**(1/3)*10**10;    #lattice constant(angstrom)

#Result
print "lattice constant is",round(a,3),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
N=6.02*10**26;           #Avagadro Number
rho=7870;    #density(kg/m**3)
M=55.85;    #atomic weight(amu)
a=2.9*10**-10;    #lattice constant(m)

#Calculation
n=a**3*rho*N/M;      #number of atoms

#Result
print "number of atoms is",int(n)

#importing modules
import math
from __future__ import division

#Variable declaration
N=6.02*10**26;           #Avagadro Number
M=63.5;    #atomic weight(amu)
r=0.1278*10**-9;    #atomic radius(m)
n=4;

#Calculation
a=r*math.sqrt(8);    #lattice constant(m)
rho=n*M/(N*a**3);      #density(kg/m**3)

#Result
print "density is",round(rho,2),"kg/m**3"
print "answer in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
r1=1.258*10**-10;     #radius(m)
r2=1.292*10**-10;     #radius(m)

#Calculation
a_bcc=4*r1/math.sqrt(3);
v=a_bcc**3;
V1=v/2;
a_fcc=2*math.sqrt(2)*r2;
V2=a_fcc**3/4;
V=(V1-V2)*100/V1;           #percent volume change is",V,"%"

#Result
print "percent volume change is",round(V,1),"%"

#importing modules
import math
from __future__ import division
from sympy import Symbol

#Variable declaration
r=Symbol('r')

#Calculation
a=4*r/math.sqrt(2);
R=(4*r/(2*math.sqrt(2)))-r;

#Result
print "maximum radius of sphere is",round(R/r,3),"r"

#importing modules
import math
from __future__ import division

#Variable declaration
N=6.023*10**23;           #Avagadro Number
Mw=23+35.5;        #molecular weight of NaCl
rho=2.18;    #density(gm/cm**3)

#Calculation
M=Mw/N;        #mass of 1 molecule(gm)
Nv=rho/M;      #number of molecules per unit volume(mole/cm**3)
Na=2*Nv;     #number of atoms
a=(1/Na)**(1/3)*10**8;   #distance between atoms(angstrom)

#Result
print "distance between atoms is",round(a,2),"angstrom"

