from scipy.optimize import fsolve
from math import radians, degrees, pi, cos, sin

#variable declaration
a=6;
sigma_s=60000.0;
su_s=91000.0;
sigma_c=10000.0;
su_c=30000;
a=radians(a);


#calculation
def s(fi):
    return cos(fi-a)*sin(fi)-sigma_s/su_s*(cos(pi/4-a/2)*sin(pi/4+a/2))
def c(fi):
    return cos(fi-a)*sin(fi)-sigma_c/su_c*(cos(pi/4-a/2)*sin(pi/4+a/2))
fi1=fsolve(s,0);
fi2=fsolve(c,0);
fi1=degrees(fi1);
fi2=degrees(fi2);

#result
print('\nShear Plane Angle for 1040 steel= %g deg')%(fi1);
print('\nShear Plane Angle for Copper = %g deg')%(fi2);



from math import sin
from math import cos
from math import tan
from math import atan
from math import radians
from math import sqrt
from math import degrees

#variable declaration
v=500;
alpha=6;
b=0.4;
t=0.008;
Fv=100;
Fh=250;
L=20;
rho=0.283;
m=13.36;
m=m/454;            #conversion to lb

#calculation
tc=m/(rho*b*L);
r=t/tc;
alpha=radians(alpha);
fi=atan(r*cos(alpha)/(1-r*sin(alpha)));
#fi=degrees(fi);
mu=(Fv+Fh*tan(alpha))/(Fh-Fv*tan(alpha));
be=atan(mu);
Pr=sqrt(Fv**2+Fh**2);
Ft=Pr*sin(be);
p_fe=Ft*r/Fh;
Fs=Fh*cos(fi)-Fv*sin(fi);
vs=v*cos(alpha)/cos(fi-alpha);
p_se=Fs*vs/(Fh*v);
U=Fh*v/(b*t*v);
U=U/33000;                     #conversion to hp
U=U/12;                         #conversion of ft units to in units
fi=degrees(fi);

#result
print('\nSlip plane angle = %g deg\nPercentage of total energy that goes into friction = %g percent\nPercentage of total energy that goes into shear = %g percent\nTotal energy per unit volume = %g hp min/in^3')%(fi,p_fe*100,p_se*100,U);



#variable declaration
d=0.5;

#calculation
t1=(1/d)**(1/0.12);
t2=(1/d)**(1/0.3);

#result
print('\nFor High Speed steel tool, increase in tool life is given by: t2 = %g t1')%(t1);
print('\nFor Cemented carbide tool, increase in tool life is given by: t2 = %g t1')%(t2);



#variable declaration
U=40;
uw=0.3;
b=1.2;
v=30;
d=0.05;

#calculation
b=b*10**-3;                   #conversion to m
d=d*10**-3;                     #conversion to m
U=U*10**9;                     #conversion to Pa
M=uw*b*d;
P=U*M;
F=P/v;

#result
print('Tangential force = %g N')%(F);



