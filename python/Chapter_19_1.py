from math import pi
from math import radians
from math import tan
from math import log

#variable declaration
def cot(x):
    return 1/tan(x);
Ab=10;
r=0.2;
alpha=12;
mu=0.09;
n=0.3;
K=1300;
v=3;

#calculation
alpha=radians(alpha);
B=mu*cot(alpha/2);
e1=log(1/(1-r));
sigma=K*e1**0.3/(n+1);
Aa=Ab*(1-r);
sigma_xa=sigma*((1+B)/B)*(1-(Aa/Ab)**B);
Aa=pi*Aa**2/4;
Pd=sigma_xa*Aa;
Pd=Pd/1000;                        #conversion to kilo units
P=Pd*v;
H=P/0.746;

#result
print('\nDrawing Stress = %g MPa\nDrawing Force = %g kN\nPower = %g kW\nHorsepower = %g hp')%(sigma_xa,Pd,P,H);

from math import radians
from math import tan
from math import log

#variable declaration
def cot(x):
    return 1/tan(x);
alpha=12;
r=0.2;
mu=0.09;
n=0.3;
K=1300;
v=3;

#calculation
alpha=radians(alpha);
B=mu*cot(alpha/2);
e1=log(1/(1-r));
sigma_xa=K*e1**0.3/(n+1);
r1=1-((1-(B/(B+1)))**(1/B));
e=log(1/(1-r1));
sigma0=1300*e**0.3;
r2=1-(1-((sigma0/sigma_xa)*(B/(B+1)))**(1/B));

#result
print('\nBy First Approximation, r = %g\nBy Second Approximation, r = %g')%(r1,r2);

