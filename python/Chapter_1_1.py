#variable declaration
from math import pi
from math import sqrt
y_b=2;
G=75;
G=G*10**9;       #conversion to Pa
L=0.01;
L=L*10**-3;       #conversion to m
nu=0.3;

#calculation
T=sqrt((3*pi*y_b*G)/(8*(1-nu)*L));
T=T/10**6;

#result
print ('Shear Stress Required to nucleate a grain boundary crack in high temperature deformation = %g MPa') %(T)

