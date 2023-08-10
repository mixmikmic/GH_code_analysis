#importing modules
import math
from __future__ import division

#Variable declaration
V=475;     #volume(m**3)
aw=200;    #area of wall(m**2)
ac=100;    #area of ceiling(m**2)
ac_w=0.025;    #absorption coefficient of wall
ac_c=0.02;    #absorption coefficient of ceiling
ac_f=0.55;    #absorption coefficient of floor

#Calculation
sigma_as=(aw*ac_w)+(ac*ac_c)+(ac*ac_f);     
T=0.165*V/sigma_as;          #reverberation time of hall(s)

#Result
print "reverberation time of hall is",round(T,3),"s"

#importing modules
import math
from __future__ import division

#Variable declaration
V=12500;     #volume(m**3)
T1=1.5;      #reverberation time(sec)
n=200;    #number of cushioned chairs

#Calculation
sigma_as=0.165*V/T1;    
T2=0.165*V/(sigma_as+n);     #new reverberation time(s)

#Result
print "new reverberation time is",round(T2,2),"s"

#importing modules
import math
from __future__ import division

#Variable declaration
V=5000;    #volume(m**3)
T=1.25;    #time(s)

#Calculation
sigma_as=0.165*V/T;          #total absorption in the hall(OWU)

#Result
print "total absorption in the hall is",sigma_as,"OWU"

#importing modules
import math
from __future__ import division

#Variable declaration
V=9500;    #volume(m**3)
T=1.5;    #time(s)
x=100;    #absorption(sabines)

#Calculation
sigma_as=0.165*V/T;          #total absorption in the hall(OWU)
T=0.165*V/(sigma_as+x);      #new period of reverberation(s)

#Result
print "total absorption in the hall is",sigma_as,"OWU"
print "new period of reverberation is",round(T,3),"s"

#importing modules
import math
from __future__ import division

#Variable declaration
V=20*15*5;    #volume(m**3)
T=3.5;    #time(s)
A=950;    #surface area(m**2)

#Calculation
sigma_as=0.165*V/T;          #total absorption in the hall(OWU)
ac=sigma_as/A;       #average absorption coefficient

#Result
print "total absorption in the hall is",round(sigma_as,3),"OWU"
print "average absorption coefficient is",round(ac,3),"sabine/m**2"

#importing modules
import math
from __future__ import division

#Variable declaration
V=2265;    #volume(m**3)
sigma_as=92.9;    #absorption(m**2)
a=18.6;      #area(m**2)

#Calculation
T=0.165*V/sigma_as;          #reverberation time of hall(s)
T1=0.165*V/2;      
inc=T1-sigma_as;       #increase in absorption(OWU)
n=inc/a;        #number of persons to be seated

#Result
print "reverberation time of hall is",round(T,3),"s"
print "number of persons to be seated is",int(n)

