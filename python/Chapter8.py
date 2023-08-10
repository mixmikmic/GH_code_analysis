#importing modules
import math
from __future__ import division

#Variable declaration
k=10**4;     #Force constant(dyne/cm)
x=5;         #displacement(cm)
m=100;       #mass(gm)
R=100;       #resistance(dyne/cm)
At=1;      #amplitude(cm)
A0=5;      #amplitude(cm)

#Calculation
E=(1/2)*k*x**2;   #restoration energy(erg)
v=1/(2*math.pi)*math.sqrt(k/m)  #frequency(Hz)
b=R/(2*m);          
t=math.log(A0/At)/b;      #time taken for reduction of amplitude(sec)

#Result
print "restoration energy is",int(E),"erg"
print "frequency is",int(v*math.pi),"/math.pi Hz"
print "time taken for reduction of amplitude is",round(t,2),"sec"

#importing modules
import math
from __future__ import division

#Variable declaration
new=300;         #frequency(Hz)
EbyE0=1/10;      #ratio of energy
Q=5*10**4;       #Q factor

#Calculation
tbytow=math.log(1/EbyE0);
tow=Q/(2*math.pi*new);   
t=tbytow*tow;      #time taken(sec)

#Result
print "time taken is",round(t,2),"sec"
print "answer in the book varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
Q=2.2*10**3;       #Q value of sonometer wire
new=210;           #frequency(Hz)

#Calculation
tow=Q/(2*math.pi*new);     #torque(Nm)
t=4*tow;           #time taken(sec)

#Result
print "time taken is",int(t),"sec"
print "procedure followed in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
m=0.5;     #mass(kg)
g=9.8;     #acceleration due to gravity(m/sec**2)
x=0.05;    #displacement(m)

#Calculation
k=m*g/x;     
omega0=math.sqrt(k/m);    #angular velocity
T=50*2*math.pi/omega0;       #time taken for 50 oscillations(sec)
b=math.log(4)/T;      #damping factor(N/m)
R=2*b*m;     #resistance(ohm)
Q=m*omega0/R;     #Q-factor

#Result
print "damping factor is",round(b,4),"N/m"
print "Q-factor is",round(Q,1)
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
m=1;     #mass(gm)
R=10;     #damping constant
E=50;     #energy(J)
E0=200;   #energy(J)
new=200;    #frequency(Hz)

#Calculation
b=R/(2*m);
t=math.log(E0/E)/(2*b);     #time taken(sec)
n=new*t;          #number of oscillations

#Result
print "number of oscillations is",round(n,2)
print "answer in the book varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
m=0.3;     #mass(kg)
new=2;    #frequency(Hz)
Q=60;     #Q-factor

#Calculation
omega=2*math.pi*new;   #angular velocity
R=m*omega/Q;       #mechanical resistance(m/sec)
b=R/m;       #damping constant
k=4*(math.pi**2)*m;    #spring constant(N/cm)

#Result
print "mechanical resistance is",round(R,4),"m/sec"
print "damping constant is",round(b,3)
print "spring constant",round(k,2),"N/cm"
print "answer for spring constant given in the book is wrong"

