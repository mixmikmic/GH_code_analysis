from math import log

#For Bar which is double in length
#variable declaration 1
L2=2;
L1=1;

#calculation 1
e=(L2-L1)/L1;
e1=log(L2/L1);
r=1-L1/L2;

#result 1
print('\nEnginering Strain = %g\nTrue Strain = %g\nReduction = %g')%(e,e1,r);



#For bar which is halved in length
#variable declaration 2
L1=1;
L2=0.5;

#calculation 2
e=(L2-L1)/L1;
e1=log(L2/L1);
r=1-L1/L2;

#result 2
print('\n\nEnginering Strain = %g\nTrue Strain = %g\nReduction = %g')%(e,e1,r);


from scipy.integrate import quad
from math import log

#variable declaration
D0=25.0;
D1=20.0;
D2=15.0;
def integrand(e):
    return 200000*e**0.5

#calculation
ep1=log((D0/D1)**2);
U1,U1_err=quad(integrand,0,ep1);
ep2=log((D1/D2)**2);
U2,U2_err=quad(integrand,ep1,ep1+ep2);

#result
print('\nPlastic work done in 1st step = %g lb/in^2\nPlastic work done in 2nd step = %g lb/in^2\n')%(U1,U2);


from math import sin
from math import radians

#variable declaration
alpha=60;

#calculation
r=radians(alpha);
mu=1/sin(r);
p_2k=mu*5/2;

#result
print('Pressure  = %g')%(p_2k);



#variable declaration
Al_s=200;
Al_e=1;
Al_p=2.69;
Al_c=0.215;
Ti_s=400;
Ti_e=1;
Ti_p=4.5;
Ti_c=0.124;
J=4.186;
b=0.95;

#calculation
Al_Td=Al_s*Al_e*b/(Al_p*Al_c*J);
Ti_Td=Ti_s*Ti_e*b/(Ti_p*Ti_c*J);

#result
print('\nTemperature Rise for aluminium = %g C\nTemperature Rise for titanium = %g C\n')%(Al_Td,Ti_Td);



from math import sqrt

#variable declaration
Do=60;
Di=30;
def1=70;
def2=81.4;
h=10;
a=30;

#calculation1
di=sqrt((Do**2-Di**2)*2-def1**2);
pr=(Di-di)/Di*100;
m=0.27;
p_s=1+2*m*a/(sqrt(3)*h);

#result 1
print('\nFor OD after deformation being 70 mm, Di = %g mm\nPrecent change in inside diameter = %g percent\nPeak pressure = %g')%(di,pr,p_s);

#calculation 2
di=sqrt(def2**2-(Do**2-Di**2)*2);
pr=(Di-di)/Di*100;
m=0.05;
p_s=1+2*m*a/(sqrt(3)*h);

#result 2
print('\n\n\n\nFor OD after deformation being 81.4 mm, Di = %g mm\nPrecent change in inside diameter = %g percent\nPeak pressure = %g')%(di,pr,p_s);

