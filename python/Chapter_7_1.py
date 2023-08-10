from math import sqrt

#variable declaration
E=95;
E=E*10**9;
Ys=1000;
Ys=Ys*10**-3;                 #conversion to J/m^2
a0=1.6;
a0=a0*10**-10;                   #conversion to m

#calculation
sigma_max=sqrt(E*Ys/a0)
sigma_max=sigma_max*10**-9;

#result
print('Cohesive strength of a silica fiber = %g GPa')%(sigma_max);

from math import sqrt

#variable declaration
E=100;
E=E*10**9;
Ys=1;
a0=2.5*10**-10;
c=10**4*a0;

#calculation
sigma_f=sqrt(E*Ys/(4*c));
sigma_f=sigma_f*10**-6;

#result
print('Fracture Stress = %g MPa')%(sigma_f);

