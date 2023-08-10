#variable declaration
from math import pi
G=40;
G=G*10**9;          #conversion to N/m^2
b=2.5;
b=b*10**-10;          #conversion to m
r=1200;
r=r*10**-10;          #conversion to m
l=0.04;
l=l*10**-3;            #conversion to m

#calculation
F=G*b**2/(2*pi*r);
Ft=F*l;

#result
print('The Total force on the dislocation is = %g N')%(Ft);

