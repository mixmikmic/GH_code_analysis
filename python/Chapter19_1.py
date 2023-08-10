#importing modules
import math
from __future__ import division

#Variable declaration
s=3.5;      #speed(km/h)
t=12/60;    #time(hr)

#Calculation
d=s*t*1000;     #distance covered(m)

#Result
print "distance covered is",d,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
d=124;     #distance(km)
s=45;      #speed(km/h)

#Calculation
t=d/s;      #time(hr)
tm=(t-int(t))*60;     #time(minutes)

#Result
print "time taken is",int(t),"hour",int(tm),"minutes"    

#importing modules
import math
from __future__ import division

#Variable declaration
d=360*1000;     #distance(m)
v=20;      #velocity(m/s)

#Calculation
t=d/v;     #time taken(min)

#Result
print "time taken is",t/(60*60),"hours"

#importing modules
import math
from __future__ import division

#Variable declaration
n=4;    #number of rests
tr=5;    #time for each rest(min)
d=5;    #distance(km)
s=10;   #speed(km/hr)

#Calculation
rt=n*tr;   #rest time(min)
t=(d*60/s)+rt;   #total time taken(min)

#Result
print "total time taken is",t,"minutes"

#importing modules
import math
from __future__ import division

#Variable declaration
abyb=5/7;    #factor
deltat=6;    #change in time(min)

#Calculation
bbya=1/abyb;      
t=deltat/(bbya-1);     #usual time(minutes)

#Result
print "usual time is",t,"minutes"

#importing modules
import math
from __future__ import division

#Variable declaration
abyb=7/6;    #factor
deltat=4;    #change in time(min)

#Calculation
bbya=1/abyb;      
t=deltat/(1-bbya);     #usual time(minutes)

#Result
print "usual time is",t,"minutes"

#importing modules
import math
from __future__ import division

#Variable declaration
t=10+5;     #difference in time(min)
v1=4;    #speed(km/h)
v2=5;    #speed(km/h)

#Calculation
d=v1*v2*t/60;     #distance(km)

#Result
print "distance is",d,"km"

#importing modules
import math
from __future__ import division

#Variable declaration
t1=8;     #time(hr)
t2=6+(40/60);     #time(hr)
d=5;     #distance(km)

#Calculation
v=d/((t1/t2)-1);     #slower speed of train(km/h)

#Result
print "slower speed of train is",v,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
s1=80;    #speed(km/h)
s2=60;    #speed(km/h)

#Calculation
st=(s1-s2)/s1;     #stoppage time(hr)

#Result
print "train stops at",st*60,"minutes/hour"

#importing modules
import math
from __future__ import division

#Variable declaration
x=4;     #speed(km/h)
a=30/60;    #time(hr)
y=2;    #speed(km/h)
b=20/60;    #time(hr)

#Calculation
d=(x+y)*(a+b)*a*b*x*y/((b*x)-(a*y))**2;        #distance(km)

#Result
print "distance is",d,"km"

#importing modules
import math
from __future__ import division

#Variable declaration
d=400;     #distance(m)
s1=10;     #speed(km/h)
s2=15;     #speed(km/h)

#Calculation
x=d*s1/(s2-s1);     #distance ran by theif(m)

#Result
print "distance ran by theif is",x,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
d1=450;       #distance(km)
d2=740;       #distance(km)
t1=7;         #time(hrs)
t2=10;        #time(hrs)

#Calculation
Va=(d1+d2)/(t1+t2);     #average speed(km/h)

#Result
print "average speed is",Va,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
V1=60;      #speed(km/h)
V2=40;      #speed(km/h)

#Calculation
Va=2*V1*V2/(V1+V2);     #average speed(km/h)

#Result
print "average speed is",Va,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
V1=40;      #speed(km/h)
V2=60;      #speed(km/h)
V3=70;      #speed(km/h)
t1=30/60;         #time(hrs)
t2=45/60;        #time(hrs)
t3=2;         #time(hrs)

#Calculation
Va=((V1*t1)+(V2*t2)+(V3*t3))/(t1+t2+t3);     #average speed(km/h)

#Result
print "average speed is",int(Va),"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
V1=60;      #speed(km/h)
V2=30;      #speed(km/h)
d=240;      #distance(km)
t=6;        #time(hrs)

#Calculation
Va=d/t;     #average speed(km/h)
t60=Va-V2;
t30=V1-Va;
T=t60*t/(t60+t30);      #time taken to travel(hours)

#Result
print "time taken to travel is",T,"hours"

#importing modules
import math
from __future__ import division

#Variable declaration
t=2;     #time(hrs)
v1=100;    #speed(km/h)
v2=50;     #speed(km/h)
d=170;     #distance(km)

#Calculation
x=(t*d)-(t*v1);      #distance travelled by car(km)

#Result
print "distance travelled by car is",x,"km"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
T=42;     #total time(min)
v1=4;    #speed(km/h)
v2=5;     #speed(km/h)

#Calculation
d=T/(v1+(2*v2));     #total distance(km)

#Result
print "total distance is",d,"km"

#importing modules
import math
from __future__ import division

#Variable declaration
d=840;     #distance(km)
t=2;     #time(hrs)
s=10;    #speed(km/h)

#Calculation
x=d*s/(s*t);    
y=x/(t*(t+1));
V=d*s/(t*y);      #original speed(km/h)

#Result
print "original speed is",V,"km/h"

#importing modules
import math
from __future__ import division
from fractions import Fraction

#Variable declaration
tx=4+(30/60);      #time(hrs)
ty=1+(45/60);      #time(hrs)

#Calculation
T=tx+ty;        #total time taken(hrs)
x=T-int(T);

#Result
print "total time taken is",int(T),Fraction(x),"hrs"

#importing modules
import math
from __future__ import division

#Variable declaration
rA=4;
rB=5;     #rates of A and B
t=20;     #time(min)

#Calculation
tB=rA*t;   #time taken by B(min)
tA=t*rB;   #time taken by A(min)

#Result
print "time taken by A is",tA,"min"
print "time taken by B is",tB,"min"

#importing modules
import math
from __future__ import division

#Variable declaration
t=6+(30/60);      #time(hrs)
tl=2+(10/60);      #time lost(hrs)

#Calculation
T=t-tl;        #total time taken(hrs)

#Result
print "total time taken is",round(T,2),"hrs"

