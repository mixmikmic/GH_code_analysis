from math import log
from math import radians
from math import tan
from math import sqrt
from math import pi

#variable declaration
def cot(x):
    return 1/tan(x);
Db=6;
Df=2;
L=15;
v=2;
alpha=60;
mu=0.1;

#calculations
R=Db**2/Df**2;
e=6*v*log(R)/Db
sigma=200*e**0.15;
alpha=radians(alpha);
B=mu*cot(alpha);
p_d=sigma*((1+B)/B)*(1-R**B);
p_d=abs(p_d);
t_i=sigma/sqrt(3);
p_e=p_d+4*t_i*L/Db;
p_e=p_e*145.0377;                    #conversion to psi
A=pi*Db**2/4;
P=p_e*A;
P=P*0.000453;                      #conversion to metric tons

#result
print('\nForce required for the Operation = %g metric tons\n\n\nNote: Slight calculation errors in book')%(P);

