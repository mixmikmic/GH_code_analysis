#importing modules
import math
from __future__ import division

#Variable declaration
s=60;     #speed(km/h)
Lt=110;   #length of train(m)
L1=0;    
L2=240;   #length of pplatform(m)
L3=170;   #length of train(m)
v1=6;      #speed of man(km/h)
v2=54;     #speed of another train(km/hr)
v3=80;     #speed of another train(km/hr)

#Calculation
Vt=s*5/18;     #speed of train(m/s)
V1=v1*5/18;      #speed of man(m/s)
V2=v2*5/18;      #speed of another train(m/s)
V3=v3*5/18;      #speed of another train(m/s)
t1=(Lt+L1)/Vt;   #time taken to cross a telegraph post(s)
t2=(Lt+L1)/(Vt-V1);    #time taken to cross a man running in same direction(s)
t3=(Lt+L1)/(Vt+V1);    #time taken to cross a man running in opposite direction(s) 
t4=(Lt+L2)/Vt;     #time taken to cross a platform(s)
t5=(Lt+L3)/Vt;     #time taken to cross a train(s)
t6=(Lt+L3)/(Vt-V2);    #time taken to cross another train(s)
t6m=int(t6/60);    #time(m)
t6s=t6-(t6m*60);
t7=(Lt+L3)/(Vt+V3);    #time taken to cross another train(s)

#Result
print "time taken to cross a telegraph post is",t1,"s"
print "time taken to cross a man running in same direction is",round(t2,2),"s"
print "time taken to cross a man running in opposite direction is",t3,"s"
print "time taken to cross a platform is",t4,"s"
print "time taken to cross a train is",t5,"s"
print "time taken to cross another train is",t6m,"minutes",t6s,"s"
print "time taken to cross another train is",t7,"s"

#importing modules
import math
from __future__ import division

#Variable declaration
s=54;   #speed of train(km/hr)
t1=36;    #time(s)
t2=20;    #time(s)
V1=0;
V2=0;
L2=0;

#Calculation
Vt=s*5/18;     #speed of train(m/s)
a=Vt*(t1-t2);
L1=a+L2-(V1*t1)-(V2*t2);    #length of platform(m)

#Result
print "length of platform is",L1,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
L1=0;     #length of man(m)
L2=150;   #length of platform(m)
t1=10;    #time(s)
t2=22;    #time(s)
V1=0;
V2=0;

#Calculation
a=(L1+(V1*t1))-(L2+(V2*t2));
Vt=a/(t1-t2);    #speed of train(m/s)

#Result
print "speed of train is",Vt,"m/s"

#importing modules
import math
from __future__ import division

#Variable declaration
L2=0;   #length of man(m)
t1=18;    #time(s)
t2=10;    #time(s)
V1=0;
V2=7*5/18;     #speed(m/s)
Vt=25*5/18;    #speed of train(m/s)

#Calculation
Lt=((Vt-V2)*t2)-L2;      #length of train(m)
L1=((Vt-V1)*t1)-Lt;      #length of platform(m)

#Result
print "length of train is",Lt,"m"
print "length of platform is",L1,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
L2=0;   #length of man(m)
t1=12;    #time(s)
t2=6;    #time(s)
V1=0;
V2=-9*5/18;     #speed(m/s)
Vt=36*5/18;    #speed of train(m/s)

#Calculation
Lt=((Vt-V1)*t1)-L1;      #length of train(m)
L1=((Vt-V1)*t1)-Lt;      #length of platform(m)

#Result
print "length of train is",Lt,"m"
print "length of platform is",L1,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
L1=210;   #length of tunnel(m) 
L2=122;   #length of tunnel(m)
t1=25;    #time(s)
t2=17;    #time(s)
V1=0;
V2=0;     #speed(m/s)

#Calculation
a=(L1+(V1*t1))-(L2+(V2*t2));
Vt=a/(t1-t2);    #speed of train(m/s)
Lt=((Vt-V1)*t1)-L1;      #length of train(m)

#Result
print "speed of train is",Vt,"m/s"
print "length of train is",Lt,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
L1=250;   #length of bridge(m) 
L2=130;   #length of platform(m)
Lt=150;   #length of train(m)
t1=30;    #time(s)
V1=0;
V2=0;     #speed(m/s)

#Calculation
t2=(Lt+L2)*t1/(Lt+L1);      #time taken by the train to cross the platform(s)

#Result
print "time taken by the train to cross the platform is",t2,"s"

#importing modules
import math
from __future__ import division

#Variable declaration
Lt1_Lt2=100;     #difference in length(m)
Vt1=90*5/18;    #speed of 1st train(m/s)
Vt2=45*5/18;    #speed of 2nd train(m/s)
t1=36;    #time(s)

#Calculation
t2=((Vt1*t1)-Lt1_Lt2)/Vt2;     #time taken by the second train(s)

#Result
print "time taken by the second train is",t2,"s"

#importing modules
import math
from __future__ import division

#Variable declaration
Lt=100;    #length of train(m)
V=-6;      #speed of train(m/s)
t=18/5;     #time(s)
L=0;

#Calculation
x=t*1/t;
Vt=(Lt+L+(x*V))/x;     #speed of train(m/s)

#Result
print "speed of train is",Vt,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
Lt=75;     #length of train(m)
L1=L2=0;
t1=18;     #time(s)
t2=15;     #time(s)
V1=6*5/18;   #speed(km/h)

#Calculation
a=(Lt+L1)/t1;
b=(Lt+L2)/t2;
V2=(a+V1-b)*18/5;      #speed of second person(km/h)

#Result
print "speed of second person is",V2,"km/h"

#importing modules
import math
from __future__ import division

#Variable declaration
L1=130;   #length of train(m)
L2=110;   #length of train(m)
t1=3;
t2=60;   #time(s)

#Calculation
s1=((L1+L2)/2)*((1/t1)+(1/t2));    #speed of faster train(m/s)
s2=((L1+L2)/2)*((1/t1)-(1/t2));    #speed of slower train(m/s)

#Result
print "speed of faster train is",s1,"m/s"
print "speed of slower train is",s2,"m/s"

