#importing modules
import math
from __future__ import division

#Variable declaration
l=20;     #length of cuboid(m)
b=10;     #breadth of cuboid(m)
h=8;      #height of cuboid(m)

#Calculation
d=math.sqrt(l**2+b**2+h**2);     #length of diagonal(m)
S=2*((l*b)+(b*h)+(l*h));         #surface area(m**2)
V=l*b*h;      #volume(m**3)

#Result
print "length of diagonal is",round(d,2),"m"
print "surface area is",S,"m**2"
print "volume is",V,"m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
e=12;     #edge(m)

#Calculation
d=e*math.sqrt(3);     #length of diagonal(m)
S=6*(e**2);         #surface area(m**2)
V=e**3;      #volume(m**3)

#Result
print "length of diagonal is",round(d,2),"m"
print "surface area is",S,"m**2"
print "volume is",V,"m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
S=726;      #surface area(m**2)

#Calculation
V=(math.sqrt(S/6))**3;      #volume(m**3)

#Result
print "volume is",V,"m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
d=34.64;       #length of diagonal(m)

#Calculation
V=(d/math.sqrt(3))**3;      #volume(m**3)

#Result
print "volume is",round(V),"m**3"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
l=115;     #length of box(cm)
b=75;     #breadth of box(cm)
h=35;      #height of box(cm)
t=2.5;     #thickness(cm)

#Calculation
IV=l*b*h;      #internal volume(cm**3)
EV=(l+(2*t))*(b+(2*t))*(h+(2*t));      #external volume(cm**3)
V=EV-IV;       #volume of wood(cm**3)

#Result
print "volume of wood is",V,"cm**3"

#importing modules
import math
from __future__ import division

#Variable declaration
V=1;      #volume(m**3)
A=10000;   #area(m**2)

#Calculation
x=V*100/A;    #thickness(cm)

#Result
print "thickness is",x,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
V=54*44*10;     #volume of reservoir(m**3)
R=3/100;        #radius(m)
r=20;     #empty rate(m)

#Calculation
A=math.pi*R**2;    #area of pipe(m**2)
t=V/(A*r);      #time to empty(sec)

#Result
print "time to empty is",round(t/3600,2),"hours"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
V=3000;      #volume of water(m**3)
A=500*300;   #surface area(m**2)

#Calculation
x=V*100/A;    #depth of rain(cm)

#Result
print "depth of rain is",x,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
a=3;
b=4;
c=5;     #sides of a triangle(m)
h=10;    #height of prism(m)

#Calculation
s=(a+b+c)/2;     #semi perimeter(m)
A=math.sqrt(s*(s-a)*(s-b)*(s-c));    #base area(m**2)
V=A*h;      #volume(m**3)

#Result
print "base area is",A,"m**2"
print "volume is",V,"m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
a=7;     #base of triangle(m)
h=24;    #height(m)

#Calculation
A=math.sqrt(3)*(a**2)/4;     #base area(m**2)
V=A*h;    #volume(m**3)

#Result
print "volume is",int(V),"m**3"

#importing modules
import math
from __future__ import division

#Variable declaration
h=300*10**3;      #height(m)
d=1/9*10**-2;     #diameter(m)
w=270;     #weight of copper wire(kg)
v=0.027;   #per m**3

#Calculation
A=math.pi*(d**2)/4;    #area(m**2)
V=A*h;      #volume of wire(m**3)
W=V*w/v;    #weight of wire(kg) 

#Result
print "weight of wire is",round(W),"kg"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
r1=6/2;     #radius of 1 pipe(cm)
r2=3/2;     #radius of another pipe(cm)
h=1;    #assume

#Calculation
V1=math.pi*r1**2*h;    #volume of water in supply pipe(cm**3)
V2=math.pi*r2**2*h;    #volume of water in discharge pipe(cm**3)
N=V1/V2;     #number of discharge pipes

#Result
print "number of discharge pipes is",N

#importing modules
import math
from __future__ import division

#Variable declaration
di=10/100;     #internal diameter(m)
de=12/100;     #external diameter(m)
l=4;    #length(m)
w=7800;    #weight of iron(kg)

#Calculation
V=math.pi*l*(de**2-di**2)/4;    #volume of iron(m**3)
W=V*w;      #weight of iron(kg)

#Result
print "weight of iron is",round(W),"kg"

#importing modules
import math
from __future__ import division

#Variable declaration
h=10;    #height of pyramid(m)
d=10;    #length of diagonal(m)

#Calculation
A=d**2/2;    #base area(m**2)
V=A*h/3;     #volume of pyramid(m**3)
a=d/math.sqrt(2);    #side of square(m)
p=4*a;    #base perimeter(m)
x=(h**2)+(a/2)**2;     
l=math.sqrt(x);   #slant height(m)
Ls=p*l/2;   #lateral surface(m**2)

#Result
print "volume of pyramid is",round(V,2),"m**3"
print "lateral surface is",Ls,"m**2"

