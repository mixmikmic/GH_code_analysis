#importing modules
import math
from __future__ import division

#Variable declaration
d=0.282*10**-9;    #lattice spacing(m)
theta=8+(35/60);   #glancing angle(degree)
n=1;   #order
Theta=90;    #angle(degree)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
Theta=Theta*math.pi/180;    #angle(radian)
lamda=2*d*math.sin(theta)/n;    #wavelength(m)
nmax=2*d*math.sin(Theta)/lamda;    #maximum order of diffraction

#Result
print "wavelength is",round(lamda*10**10,3),"angstrom"
print "answer varies due to rounding off errors"
print "maximum order of diffraction is",round(nmax)

#importing modules
import math
from __future__ import division

#Variable declaration
d=3.04*10**-10;    #lattice spacing(m)
n=3;   #order
lamda=0.79*10**-10;    #wavelength(m)

#Calculation
theta=math.asin(n*lamda/(2*d));     #glancing angle(radian)
theta=theta*180/math.pi;          #glancing angle(degrees)

#Result
print "glancing angle is",round(theta,3),"degrees"

#importing modules
import math
from __future__ import division

#Variable declaration
a=0.28*10**-9;    #lattice spacing(m)
n=2;   #order
lamda=0.071*10**-9;    #wavelength(m)
h=1;
k=1;
l=0;

#Calculation
d110=a/math.sqrt(h**2+k**2+l**2);     #spacing(m)
theta=math.asin(n*lamda/(2*d110));    #glancing angle(radian)
theta=theta*180/math.pi;          #glancing angle(degrees)

#Result
print "glancing angle is",round(theta,2),"degrees"
print "answer in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
n=1;   #order
lamda=3*10**-10;    #wavelength(m)
h=1;
k=0;
l=0;
theta=40;    #angle(degree)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=n*lamda/(2*math.sin(theta));     #space of plane(m)
a=d*math.sqrt(h**2+k**2+l**2);     
V=a**3;      #volume of unit cell(m**3)

#Result
print "space of plane is",round(d*10**10,4),"angstrom"
print "volume of unit cell is",round(V*10**30,3),"*10**-30 m**3"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
a=3;    #lattice spacing(m)
n=1;   #order
lamda=0.82*10**-9;    #wavelength(m)
theta=75.86;    #angle(degree)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=n*10**10*lamda/(2*math.sin(theta));    #spacing(angstrom)

#Result
print "spacing is",round(d,2),"angstrom"
print "answer in the book is wrong. hence the miller indices given in the book are also wrong and cannot be calculated"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;    #charge(c)
m=9.1*10**-31;    #mass(kg)
h=6.625*10**-34;   #plank constant
n=1;   #order
theta=9+(12/60)+(25/(60*60));    #angle(degree)
V=235.2;    #kinetic energy of electron(eV)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
lamda=h*10**10/math.sqrt(2*m*e*V);   
d=n*lamda/(2*math.sin(theta));       #interplanar spacing(angstrom)

#Result
print "interplanar spacing is",round(d,3),"angstrom"
print "answer in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
n=1;   #order
h=1;
k=1;
l=1;
e=1.6*10**-19;    #charge(c)
theta=27.5;    #angle(degree)
H=6.625*10**-34;    #plancks constant
c=3*10**10;    #velocity of light(m)
a=5.63*10**-10;     #lattice constant(m)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=a/math.sqrt(h**2+k**2+l**2);
lamda=2*d*math.sin(theta)/n;      #wavelength of Xray beam(m)
E=H*c/(e*lamda);           #energy of Xray beam(eV)         

#Result
print "wavelength of X-ray beam is",int(lamda*10**10),"angstrom"
print "energy of Xray beam is",round(E/10**5,2),"*10**5 eV"
print "answer in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
e=1.6*10**-19;    #charge(c)
theta=56;    #angle(degree)
V=854;    #voltage(V)
n=1;      #order of diffraction
m=9.1*10**-31;    #mass(kg)
h=6.625*10**-34;   #plank constant

#Calculation
theta=theta*math.pi/180;    #angle(radian)
lamda=h/math.sqrt(2*m*e*V);    #wavelength(m)
d=n*lamda/(2*math.sin(theta));     #spacing of crystal(m)

#Result
print "spacing of crystal is",round(d*10**10,3),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
n=1;   #order
h=2;
k=0;
l=2;
theta=34;    #angle(degree)
lamda=1.5;   #wavelength(angstrom)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=n*lamda/(2*math.sin(theta));     #spacing of crystal(angstrom)
a=d*math.sqrt(h**2+k**2+l**2);     #lattice parameter(angstrom)

#Result
print "lattice parameter is",round(a,3),"angstrom"
print "answer in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
n=1;   #order
h=1;
k=1;
l=1;
e=1.6*10**-19;    #charge(c)
V=5000;    #voltage(V)
m=9.1*10**-31;    #mass(kg)
H=6.625*10**-34;   #plank constant
d=0.204*10**-9;    #interplanar spacing(m)

#Calculation
lamda=H/math.sqrt(2*m*e*V);    #wavelength(m)
theta=math.asin(n*lamda/(2*d));    #bragg's angle(radian)
theta=theta*180/math.pi;    #bragg's angle(degree)

#Result
print "bragg's angle is",round(theta,4),"degrees"

