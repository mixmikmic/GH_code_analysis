#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;    #charge(c)
m=9.1*10**-31;    #mass(kg)
h=6.626*10**-34;   #plank constant
E=2000;            #energy(eV)

#Calculation
lamda=h/math.sqrt(2*m*E*e);    #wavelength(m)

#Result
print "wavelength is",round(lamda*10**9,4),"nm"

#importing modules
import math
from __future__ import division

#Variable declaration
V=1600;      #potential energy of electron(V)

#Calculation
lamda=12.27/math.sqrt(V);      #wavelength(m)

#Result
print "wavelength is",lamda,"angstrom"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
me=9.1*10**-31;    #mass(kg)
h=6.62*10**-34;   #plank constant
mn=1.676*10**-27;    #mass(kg)
c=3*10**8;     #velocity of light(m/s)

#Calculation
lamda=h*10**10/math.sqrt(4*mn*me*c**2);     #de broglie wavelength(angstrom)  

#Result
print "de broglie wavelength is",round(lamda*10**4,1),"*10**-4 angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
a=2*10**-10;    #length(m)
n1=2;
n2=4;
m=9.1*10**-31;    #mass(kg)
e=1.6*10**-19;    #charge(c)
h=6.626*10**-34;   #plank constant

#Calculation
E2=n1**2*h/(8*m*e*a);      #energy of second state(eV)
E4=n2**2*h/(8*m*e*a);      #energy of fourth state(eV)

#Result
print "energy of second state is",round(E2*10**-26,5),"*10**26 eV"
print "energy of second state is",round(E4*10**-26,5),"*10**26 eV"

#importing modules
import math
from __future__ import division

#Variable declaration
V=344;    #accelerated voltage(V)
n=1;
theta=60;   #glancing angle(degrees)

#Calculation
theta=theta*math.pi/180;    #glancing angle(radian)
lamda=12.27/math.sqrt(V);
d=n*lamda/(2*math.sin(theta));    #spacing of crystal(angstrom)

#Result
print "spacing of crystal is",round(d,4),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=1.66*10**-10;    #wavelength(m)
m=9.1*10**-32;    #mass(kg)
e=1.6*10**-19;    #charge(c)
h=6.626*10**-34;   #plank constant

#Calculation
E=h**2/(4*m*e*lamda**2);   #kinetic energy(eV)
v=h/(m*lamda);      #velocity(m/s)

#Result
print "kinetic energy is",round(E,2),"eV"
print "velocity is",round(v*10**-6,2),"*10**5 m/s"

#importing modules
import math
from __future__ import division

#Variable declaration
a=1*10**-10;    #length(m)
n2=2;
n3=3;
m=9.1*10**-31;    #mass(kg)
e=1.6*10**-19;    #charge(c)
h=6.626*10**-34;   #plank constant

#Calculation
E1=h**2/(8*m*e*a**2);
E2=n2**2*E1;      #energy of 1st excited state(eV)
E3=n3**2*E1;      #energy of 2nd excited state(eV)

#Result
print "ground state energy is",round(E1,2),"eV"
print "energy of 1st excited state is",round(E2,2),"eV"
print "energy of 2nd excited state is",round(E3,2),"eV"
print "answer in the book varies due to rounding off errors"

#importing modules
import math
from __future__ import division
from sympy import Symbol

#Variable declaration
n=Symbol('n');
a=4*10**-10;    #width of potential well(m)
m=9.1*10**-31;    #mass(kg)
e=1.6*10**-19;    #charge(c)
h=6.626*10**-34;   #plank constant

#Calculation
E1=n**2*h**2/(8*m*e*a**2);     #maximum energy(eV)

#Result
print "maximum energy is",round(E1/n**2,4),"*n**2 eV"

#importing modules
import math
from __future__ import division

#Variable declaration
delta_x=10**-8;     #length of box(m)
m=9.1*10**-31;    #mass(kg)
h=6.626*10**-34;   #plank constant

#Calculation
delta_v=h/(m*delta_x);     #uncertainity in velocity(m/s)

#Result
print "uncertainity in velocity is",round(delta_v/10**3,1),"km/s"

#importing modules
import math
from __future__ import division

#Variable declaration
me=9.1*10**-31;    #mass(kg)
mp=1.6*10**-27;    #mass(kg)
h=6.626*10**-34;   #plank constant
c=3*10**10;    #velocity of light(m/s)

#Calculation
lamda=h/math.sqrt(2*mp*me*c**2);    #de broglie wavelength(m)

#Result
print "de broglie wavelength is",round(lamda*10**10*10**5,5),"*10**-5 angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
m=1.675*10**-27;    #mass(kg)
h=6.626*10**-34;   #plank constant
E=0.04;     #kinetic energy(eV)
e=1.6*10**-19;    #charge(c)
n=1;
d110=0.314*10**-9;   #spacing(m)

#Calculation
E=E*e;       #energy(J)
lamda=h/math.sqrt(2*m*E);
theta=math.asin(n*lamda/(2*d110));     #glancing angle(radian)
theta=theta*180/math.pi;        #glancing angle(degrees)
theta_m=60*(theta-int(theta));

#Result
print "glancing angle is",int(theta),"degrees",int(theta_m),"minutes"
print "answer given in the book is wrong"

