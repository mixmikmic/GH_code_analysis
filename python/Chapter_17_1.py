
from math import atan

#variable declaration
mu1=0.08;
mu2=0.5;
R=12;

#calculation
alpha=atan(mu1);
dh1=mu1**2*R;
dh2=mu2**2*R;

#result
print('\nMaximum possible reduction when mu is 0.08 = %g in\n')%(dh1);
print('Maximum possible reduction when mu is 0.5 = %g in')%(dh2);


from math import sqrt
from math import exp

#variable declaration
h0=1.5;
mu=0.3;
D=36;
s_en=20;
s_ex=30;

#calculation
h1=h0-0.3*h0;
dh=h0-h1;
h_=(h1+h0)/2;
Lp=sqrt(D/2*dh);
Q=mu*Lp/h_;
sigma0=(s_en+s_ex)/2;
P=sigma0*(exp(Q)-1)*s_ex*Lp/Q;
Ps=sigma0*(Lp/(4*dh)+1)*s_ex*Lp;

#result
print('\nRolling Load = %g kips')%(P);
print('\nRolling Load  if sticking friction occurs = %g kips')%(Ps);



from math import sqrt
from math import exp

#variable declaration
h0=1.5;
mu=0.3;
D=36;
s_en=20;
s_ex=30;
C=3.34*10**-4;
P_=1357;

#calculation
h1=h0-0.3*h0;
dh=h0-h1;
h_=(h1+h0)/2;
R=D/2;
R1=R*(1+C*P_/(s_ex*(dh)));
Lp=sqrt(R1*dh);
Q=mu*Lp/h_;
sigma0=(s_en+s_ex)/2;
P2=sigma0*(exp(Q)-1)*s_ex*Lp/Q;
P2=P2*0.45359                         #conversion to tons
R2=R*(1+C*P2/(s_ex*(dh)));

#result
print('\nP2 = %g\nR2 = %g')%(P2,R2);


from math import sqrt
from math import pi
from math import log

#variable declaration
w=12;
hi=0.8;
hf=0.6;
D=40;
N=100;

#calculation
R=D/2;
dh=abs(hf-hi);
e1=log(hi/hf);
r=(hi-hf)/hi;
sigma=20*e1**0.2/1.2;
Qp=1.5;
P=2*sigma*w*sqrt(R*(hi-hf))*Qp/sqrt(3);
a=0.5*sqrt(R*dh);
a=a/12;                             #conversion to ft
hp=4*pi*a*P*N*1000/33000;

#result
print('\nRolling Load = %g\nHorsepower = %g')%(P,hp);

