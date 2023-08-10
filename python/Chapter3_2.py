#importing modules
import math
from __future__ import division

#Variable declaration
d=0.313;     #lattice spacing(m)
theta=7+(48/60);    #angle(degrees)
n=1;

#Calculation
theta=theta*math.pi/180;    #angle(radian)
lamda=2*d*math.sin(theta)/n;    #wavelength of X-rays(nm)
#when theta=90
n=2*d/lamda;       #maximum order of diffraction possible

#Result
print "wavelength of X-rays is",round(lamda,5),"nm"
print "answer varies due to rounding off errors"
print "when theta=90, maximum order of diffraction possible is",int(n)

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=1.5418;      #wavelength(angstrom)
theta=30;      #angle(degrees)
n=1;    #first order
h=1;
k=1;
l=1;

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=n*lamda/(2*math.sin(theta));     
a=d*math.sqrt(h**2+k**2+l**2);    #interatomic spacing(angstrom)

#Result
print "interatomic spacing is",round(a,2),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
d100=0.28;    #spacing(nm)
lamda=0.071;    #wavelength of X rays(nm)
n=2;    #second order

#Calculation
d110=round(d100/math.sqrt(2),3);     #spacing(nm)
x=n*lamda/(2*d110);
theta=math.asin(x);    #glancing angle(radian)
theta=theta*180/math.pi;     #glancing angle(degrees)

#Result
print "glancing angle is",int(theta),"degrees"

#importing modules
import math
from __future__ import division

#Variable declaration
a=0.38;     #lattice constant(nm)
h=1;
k=1;
l=0;

#Calculation
d=a/math.sqrt(h**2+k**2+l**2);     #distance between planes(nm)

#Result
print "distance between planes is",round(d,2),"nm"

#importing modules
import math
from __future__ import division

#Variable declaration
a=0.19;     #lattice constant(nm)
h=1;
k=1;
l=1;
lamda=0.058;    #wavelength of X rays(nm)
n=2;    #second order

#Calculation
d=a/math.sqrt(h**2+k**2+l**2);     #distance between planes(nm)
x=n*lamda/(2*d);
theta=math.asin(x);    #glancing angle(radian)
theta=theta*180/math.pi;     #glancing angle(degrees)

#Result
print "glancing angle is",round(theta),"degrees"

