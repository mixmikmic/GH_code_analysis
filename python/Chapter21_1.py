#importing modules
import math
from __future__ import division

#Variable declaration
ds=8;     #speed downstream(km/h)
us=2;     #speed upstream(km/h)

#Calculation
r=(ds+us)/2;     #rate in still water(km/h)
s=(ds-us)/2;     #speed of current(km/h)

#Result
print "rate in still water is",r,"km/h"
print  "speed of current is",s,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
ddn=30;     #distance downstream(km)
dup=20;     #distance upstream(km)
tdn=5;      #time downstream(hrs)
tup=5;      #time upstream(hrs)

#Calculation
s=(1/2)*((ddn/tdn)-(dup/tup));      #speed of current(km/h)

#Result
print "speed of current is",s,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
d=2;    #distance(km)
tdn=20;    #time downstream(min)
tup=15;    #time upstream(min)

#Calculation
x=(d/2)*((1/tdn)+(1/tup));     #speed in still water(km/min)
x=x*60;      #speed in still water(km/hr)
y=(d/2)*(-(1/tdn)+(1/tup));     #speed of current(km/min)
y=y*60;      #speed of current(km/hr)

#Result
print "speed in still water is",x,"km/hr"
print "speed of current is",y,"km/hr"

#importing modules
import math
from __future__ import division

#Variable declaration
r=4;       #rate in still water(km/hr)
tdn=1;     #assume

#Calculation
tup=2*tdn;   
d=(tdn+tup)/(tup-tdn);
s=r/d;      #speed of stream(km/h)

#Result
print "speed of stream is",round(s,1),"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
x=6;     #speed in still water(km/hr)
y=2;     #speed of river(km/hr)
t=3;     #time(hrs)

#Calculation
d=t*(x+y)/(1+y);     #distance(km)

#Result
print "distance is",d,"km"

#importing modules
import math
from __future__ import division

#Variable declaration
x=15;     #speed in still water(km/hr)
y=13;     #rate of current(km/hr)
t=15/60;    #time(hrs)

#Calculation
d=(x+y)*t;    #distance in downstream(km) 

#Result
print "distance in downstream is",d,"km"

#importing modules
import math
from __future__ import division

#Variable declaration
x=4.5;     #speed in still water(km/hr)
y=1.5;     #rate of current(km/hr)

#Calculation
avgs=(x+y)*(x-y)/x;     #average speed for total journey(km/h)

#Result
print "average speed for total journey is",avgs,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
T=55;       #time(min)
c=60;      #conversion factor from min to h
y=2;       #speed of stream(km/h)
d=10;      #distance upstream(km)

#Calculation
a=T/5;    #coefficient of x**2
b=-c*y**2;    #coefficient of x
c=-a*y**2;     #constant
x=(b**2)-(4*a*c);
x1=-b+math.sqrt(x)/(2*a);
x2=-b-math.sqrt(x)/(2*a);

#Result
print "speed of rowing in still water is",int(x2/10)

#importing modules
import math
from __future__ import division

#Variable declaration
s=10;   #speed in still water(km/h)
td=91;    #total distance(km)
t=20;    #time(h)

#Calculation
d=t/s;    #distance(km)
y2=(s**2)-td;     
y=math.sqrt(y2);      #flow of river(km/h)

#Result
print "flow of river is",y,"km/h"

