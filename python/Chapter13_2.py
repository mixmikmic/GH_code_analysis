#importing modules
import math
from __future__ import division

#Variable declaration
n1=1.55;     #refractive index of core
n2=1.50;     #refractive index of cladding

#Calculation
NA=math.sqrt(n1**2-n2**2);    #numerical aperture

#Result
print "numerical aperture is",round(NA,3)

#importing modules
import math
from __future__ import division

#Variable declaration
n1=1.563;     #refractive index of core
n2=1.498;     #refractive index of cladding

#Calculation
NA=math.sqrt(n1**2-n2**2);    #numerical aperture
alpha_i=math.asin(NA);      #angle of acceptance(radian)
alpha_i=(alpha_i*180/math.pi);    #angle(degrees)
alpha_id=int(alpha_i);
alpha_im=60*(alpha_i-alpha_id);

#Result
print "angle of acceptance is",alpha_id,"degrees",round(alpha_im,1),"minutes"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
NA=0.39;       #numerical aperture
delta=0.05;    #difference of indices

#Calculation
n1=NA/math.sqrt(2*delta);     #refractive index of core

#Result
print "refractive index of core is",round(n1,4)
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
n1=1.563;     #refractive index of core
n2=1.498;     #refractive index of cladding

#Calculation
delta=(n1-n2)/n1;    #fractional index change

#Result
print "fractional index change is",round(delta,4)

#importing modules
import math
from __future__ import division

#Variable declaration
n1=1.48;     #refractive index of core
n2=1.45;     #refractive index of cladding

#Calculation
NA=math.sqrt(n1**2-n2**2);    #numerical aperture
alpha_i=math.asin(NA);      #angle of acceptance(radian)
alpha_i=(alpha_i*180/math.pi);    #angle(degrees)
alpha_id=int(alpha_i);
alpha_im=60*(alpha_i-alpha_id);

#Result
print "numerical aperture is",round(NA,4)
print "angle of acceptance is",alpha_id,"degrees",round(alpha_im),"minutes"

#importing modules
import math
from __future__ import division

#Variable declaration
Pout=40;    #power(mW)
Pin=100;    #power(mW)

#Calculation
al=-10*math.log10(Pout/Pin);    #attenuation loss(dB)

#Result
print "attenuation loss is",round(al,2),"dB"

