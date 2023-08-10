#importing modules
import math
from __future__ import division

#Variable declaration    
V=7500;      #volume(m**3)
T=1.5;       #time(sec)

#Calculations
aS=0.165*V/T;     #total absorption in hall(OWU)

#Result
print "total absorption in hall is",aS,"OWU"

#importing modules
import math
from __future__ import division

#Variable declaration    
V=1500;      #volume(m**3)
A1=112;      #area of plastered walls(m**2)
A2=130;      #area of wooden floor(m**2)
A3=170;      #area of plastered ceiling(m**2)
A4=20;       #area of wooden door(m**2)
n=100;     #number of cushioned chairs
A5=120;    #area of audience(m**2)
C1=0.03;    #coefficient of absorption in plastered walls
C2=0.06;    #coefficient of absorption in wooden floor
C3=0.04;    #coefficient of absorption in plastered ceiling
C4=0.06;    #coefficient of absorption in wooden door
C5=1.0;     #coefficient of absorption in cushioned chairs
C6=4.7;     #coefficient of absorption in audience

#Calculations
a1=A1*C1;    #absorption due to plastered walls
a2=A2*C2;    #absorption due to wooden floor
a3=A3*C3;    #absorption due to plastered ceiling
a4=A4*C4;    #absorption due to wooden door
a5=n*C5;     #absorption due to cushioned chairs
a6=A5*C6;    #absorption due to audience 
aS=a1+a2+a3+a4+a5;       #total absorption in hall
T1=0.165*V/aS;       #reverberation time when hall is empty(sec)
T2=0.165*V/(aS+a6);    #reverberation time with full capacity of audience(sec)
T3=0.165*V/((n*C6)+aS);    #reverberation time with audience in cushioned chairs(sec)

#Result
print "reverberation time when hall is empty is",round(T1,2),"sec"
print "reverberation time with full capacity of audience is",round(T2,3),"sec"
print "reverberation time with audience in cushioned chairs is",round(T3,2),"sec"

#importing modules
import math
from __future__ import division

#Variable declaration    
V=1200;      #volume(m**3)
a1=220;      #area of wall(m**2)
a2=120;      #area of floor(m**2)
a3=120;      #area of ceiling(m**2)
C1=0.03;    #coefficient of absorption in wall
C2=0.80;    #coefficient of absorption in floor
C3=0.06;    #coefficient of absorption in ceiling

#Calculations
A1=a1*C1;    #absorption due to plastered walls
A2=a2*C2;    #absorption due to wooden floor
A3=a3*C3;    #absorption due to plastered ceiling
aS=a1+a2+a3;       #total absorption in hall
abar=(A1+A2+A3)/aS;    #average sound absorption coefficient
AS=abar*aS;       #total absorption of room(metric sabines)
T=0.165*V/AS;       #reverberation time(sec)

#Result
print "average sound absorption coefficient is",round(abar,2)
print "reverberation time is",round(T,1),"sec"

#importing modules
import math
from __future__ import division

#Variable declaration    
I0=10**-12;      #standard intensity level(watt/m**2)
A=1.4;     #area(m**2)
il=60;     #intensity level(decibels)

#Calculations
x=10**(il/10);
I=x*10**-12;      #intensity level(watt/m**2)
Ap=I*A;      #acoustic power(watt)

#Result
print "acoustic power is",Ap,"watt"
print "answer in the book is wrong"

