from math import sqrt
from __future__ import division
#(1) frequency
freque=100*10**3*sqrt(2**(1/3)-(1))
frequ2=100*10**3/sqrt(2**(1/3)-(1))
print "frequency1   =   %0.2f"%((freque)),"hertz"
print "frequency2   =   %0.2f"%((frequ2)),"hertz"
#(2)frequency
freq11=100*10**6##hertz
freq12=150*10**6##hertz
freq13=200*10**6##hertz
freq21=100*10**3##hertz
freq22=150*10**3##hertz
freq23=200*10**3##hertz
frequ1=sqrt(freq11**2+freq12**2+freq13**2)
print "frequency   =   %0.2f"%((frequ1)),"hertz"##correction in the book 269.25mega hertz
frequ1=1/sqrt((1/(freq21**2))+(1/(freq22**2))+(1/(freq23**2)))
print "frequency   =   %0.2f"%((frequ1)),"hertz"##correction in the book

freque=60##hertz
frequ1=freque*0.484
cb=1/(frequ1*2*3.14*10**3)
print "coupling capacitance   =   %0.2e"%((cb)),"/r`"

g=10*10**-3##ampere per volt
rd=5.5*10**3##ohm
rg=1*10**6##ohm
#(1) cb frequency 1decibel to 10hertz
ri=rg
r1=(rd*8*10**3)/(rd+8*10**3)
cb=10**-6/(3.14*5.07)
print "cb   =   %0.2e"%((cb)),"farad"
#(2) cb
cb=(cb*(5)/(3.52))
print "cb   =   %0.2e"%((cb)),"farad"
#(3) gain
a1=g**2*(3.26**2)
print "gain of each stage   =   %0.2e"%((a1))
#correction required in the book

freque=40*10**3##hertz
frequ1=freque/0.507
print "upper frequency   =   %0.2f"%((frequ1)),"hertz"
frequ1=freque/1.96
print "lower frequency   =   %0.2f"%((frequ1)),"hertz"

from math import log10
g=2.6*10**-3##ampere per volt
rd=7.7*10**3##ohm
rd1=12*10**3##ohm
cb=0.005*10**-6##farad
#(1) voltage gain
volgai=g*((1/rd)+1/rd1+1/(1*10**3))
volgai=(20*(log10(10.8)))*3
print "overal voltage gain   =   %0.2f"%((volgai)),"decibel"##correction in the book
#(2) lower frequency
r=rd*rd1/(rd+rd1)
freque=1/((2*3.14)*(r+1*10**6)*cb)
print "lower frequency of each   =   %0.2f"%((freque)),"hertz"
#(3) overal lower frequency
freque=freque*1.96
print "lower frequency overal   =   %0.2f"%((freque)),"hertz"

hfe=50
hie=1.1*10**3##ohm
#(1) gain
r1=2*10**3##ohm
volgai=-hfe*r1/(hie)
r11=25*10**3*hie/(25*10**3+hie)
r11=r1*r11/(r1+r11)
volga1=-hfe*r11/hie
volgai=volgai*volga1
print "voltage gain   =   %0.2f"%((volgai))
freque=20##hertz
ri=25*10**3*hie/(25*10**3+hie)
cb=1/(2*3.14*(ri+r1)*(freque))
print "cb   =   %0.2e"%((cb)),"farad"
cb=1/(2*3.14*3.05*10**3*10/3.14)
print "cb   <=   %0.2e"%((cb)),"farad"

from math import atan, degrees
theta1=degrees(atan(0.1))
print "theta1   =   %0.2f"%((theta1))
print "phase constant 10f1<=f<=0.1f11"

