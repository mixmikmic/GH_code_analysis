
from math import sqrt
from math import pi
from math import cos

#variable declaration
a=5;
a=a*10**-3;             #conversion to m
t=1.27;          #in cm
t=t*10**-2;        #conversion to m
def sec(x):
    return 1/cos(x);

#calculation
K_Ic=24;
sigma=K_Ic/(sqrt(pi*a)*sqrt(sec(pi*a/(2*t))));

#result
print('Since Fracture Toughness of the material is = %g MPa\n and the applied stress is 172 MPa thus the flaw will propagate as a brittle fracture')%(sigma);

from math import pi

#variable declaration
K_Ic=57;
sigma0=900;
sigma=360;
Q=2.35;

#calculation
a_c=K_Ic**2*Q/(1.21*pi*sigma**2);
a_c=a_c*1000;                         #conversion to mm

#result
print('\nCritical Crack depth = %g mm\nwhich is greater than the thickness of the vessel wall, 12mm')%(a_c);

from math import sqrt
from math import pi

#variable declaration
a=10;
a=a*10**-3;                 #conversion to m
sigma=400;
sigma0=1500;

#calculation
rp=sigma**2*a/(2*pi*sigma0**2);
rp=rp*1000;                      #conversion to mm
K=sigma*sqrt(pi*a);
K_eff=sigma*sqrt(pi*a)*sqrt(a+pi*rp);

#result
print('\nPlastic zone size = %g mm\nStress Intensity Factor = %g MPa m^(1/2)\n\n\nNote: Calculation Errors in book')%(rp,K_eff);

