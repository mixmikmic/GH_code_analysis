#importing modules
import math
from __future__ import division

#Variable declaration
lamda=0.071*10**-9;     #wavelength(m)
a=0.28*10**-9;    #lattice constant(m)
h=1;
k=1;
l=0;
n=2;    #order of diffraction

#Calculation
d=a/math.sqrt(h**2+k**2+l**2);
x=n*lamda/(2*d);     
theta=math.asin(x);     #angle(radian)
theta=theta*180/math.pi;    #glancing angle(degrees)

#Result
print "glancing angle is",int(theta),"degrees"

#importing modules
import math
from __future__ import division

#Variable declaration
n=1;    #order of diffraction
theta1=8+(35/60);    #angle(degrees)
d=0.282;     #spacing(nm)
theta2=90;

#Calculation
theta1=theta1*math.pi/180;    #angle(radian)
lamda=2*d*math.sin(theta1)/n;    #wavelength(nm)
theta2=theta2*math.pi/180;    #angle(radian)
nmax=2*d/lamda;     #maximum order of diffraction

#Result
print "wavelength is",round(lamda,4),"nm"
print "maximum order of diffraction is",round(nmax)

#importing modules
import math
from __future__ import division

#Variable declaration
T1=500+273;    #temperature(K)
T2=1000+273;   #temperature(K)
f=1*10**-10;   #fraction

#Calculation
x=round(T1/T2,5);
y=round(math.log(f),3);
w=round(x*y,3);
F=math.exp(w);    #fraction of vacancy sites

#Result
print "fraction of vacancy sites is",round(F*10**7,3),"*10**-7"

#importing modules
import math
from __future__ import division

#Variable declaration
a=1;    #assume
h1=1;
k1=0;
l1=0;
h2=1;
k2=1;
l2=0;
h3=1;
k3=1;
l3=1;

#Calculation
d100=a*6/(h1**2+k1**2+l1**2);
d110=a*6/(h2**2+k2**2+l2**2);
d111=a*(6)/(h3**2+k3**2+l3**2);

#Result
print "ratio is math.sqrt(",d100,"): math.sqrt(",d110,"): math.sqrt(",d111,")"

#importing modules
import math
from __future__ import division

#Variable declaration
n=1;    #order of diffraction
theta=38.2;    #angle(degrees)
lamda=1.54;    #wavelength(angstrom)
h=2;
k=2;
l=0;

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=n*lamda/(2*math.sin(theta));
a=d*math.sqrt(h**2+k**2+l**2);    #lattice parameter of nickel(angstrom)

#Result
print "lattice parameter of nickel is",round(a,3),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
theta=90;    #angle(degrees)
lamda=1.5;    #wavelength(angstrom)
d=1.6;    #spacing(angstrom)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
n=2*d*math.sin(theta)/lamda;    #order of diffraction

#Result
print "order of diffraction is",int(n)

#importing modules
import math
from __future__ import division

#Variable declaration
h=1;
k=1;
l=0;
d=0.203*10**-9;    #spacing(m)

#Calculation
a=d*math.sqrt(h**2+k**2+l**2);    #length of unit cell(m)
V=a**3;    #volume of unit cell(m**3)
r=math.sqrt(3)*a/4;    #radius of the atom(m)

#Result
print "length of unit cell is",round(a*10**9,3),"*10**-9 m"
print "volume of unit cell is",round(V*10**27,5),"*10**-27 m**3"
print "radius of the atom is",round(r*10**9,4),"*10**-9 m"

#importing modules
import math
from __future__ import division

#Variable declaration
theta=90;    #angle(degrees)
lamda=1.5;    #wavelength(angstrom)
d=1.6;    #spacing(angstrom)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
n=2*d*math.sin(theta)/lamda;    #order of diffraction

#Result
print "order of diffraction is",int(n)

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=0.065;    #wavelength(nm)
a=0.26;      #edge length(nm)
h=1;
k=1;
l=0;
n=2;

#Calculation
d=a/math.sqrt(h**2+k**2+l**2);      
x=n*lamda/(2*d);     
theta=math.asin(x);        #glancing angle(radian)
theta=theta*180/math.pi;    #glancing angle(degrees)
theta_d=int(theta);       
theta_m=(theta-theta_d)*60;
theta_s=(theta_m-int(theta_m))*60;

#Result
print "glancing angle is",theta_d,"degrees",int(theta_m),"minutes",int(theta_s),"seconds"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=1.54;    #wavelength(angstrom)
h=1;
k=1;
l=1;
n=1;
theta=19.2;    #angle(degrees)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=n*lamda/(2*math.sin(theta));     
a=d*math.sqrt(h**2+k**2+l**2);      #cube edge of unit cell(angstrom)

#Result
print "cube edge of unit cell is",round(a,3),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=1.54;    #wavelength(angstrom)
h=2;
k=2;
l=0;
n=1;
theta=38.2;    #angle(degrees)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=n*lamda/(2*math.sin(theta));     
a=d*math.sqrt(h**2+k**2+l**2);    #lattice parameter of nickel(angstrom)

#Result
print "lattice parameter of nickel is",round(a,3),"angstrom"

#importing modules
import math
from __future__ import division

#Variable declaration
a=0.36;      #edge length(nm)
h1=1;
k1=1;
l1=1;
h2=3;
k2=2;
l2=1;

#Calculation
d1=a/math.sqrt(h1**2+k1**2+l1**2);    #interplanar spacing for (111)(nm)
d2=a/math.sqrt(h2**2+k2**2+l2**2);    #interplanar spacing for (321)(nm)

#Result
print "interplanar spacing for (111) is",round(d1,3),"nm"
print "interplanar spacing for (321) is",round(d2,3),"nm"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=0.675;    #wavelength(angstrom)
n=3;    #order of diffraction
theta=5+(25/60);    #angle(degrees)

#Calculation
theta=theta*math.pi/180;    #angle(radian)
d=lamda/(2*math.sin(theta));   
theta3=math.asin(3*lamda/(2*d));    #glancing angle(radian)
theta3=theta3*180/math.pi;    #glancing angle(degrees)
theta_d=int(theta3);       
theta_m=(theta3-theta_d)*60;

#Result
print "glancing angle is",theta_d,"degrees",int(theta_m),"minutes"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=0.79;    #wavelength(angstrom)
n=3;    #order of diffraction
d=3.04;    #spacing(angstrom)

#Calculation
x=round(n*lamda/(2*d),4);
theta=math.asin(x);         #glancing angle(radian)
theta=theta*180/math.pi;    #glancing angle(degrees)
theta_d=int(theta);       
theta_m=(theta-theta_d)*60;
theta_s=(theta_m-int(theta_m))*60;

#Result
print "glancing angle is",theta_d,"degrees",int(theta_m),"minutes",int(theta_s),"seconds"
print "answer given in the book is wrong"

