from __future__ import division
quanti=3*10**17#
voltag=10*10**3##volt
distan=40*10**-3##metre per minute
w=quanti*1.6*10**-19*voltag
w=w/60##per second

print "power to electrons   =   ",round((w),2),"watts"

from math import sqrt
sensit=5## per centimetre
q=50*10**-6##second per centimetre
petope=5.4##centimetre
horiax=8.4##centimetre
voltag=petope*sensit#
voltag=voltag/((2)*sqrt(2))#
#one cycle
horiax=(horiax/2)*q#
freque=1/horiax#
print "input voltage   =   ",round((voltag),2),"volt"
print "frequency   =   ",round((freque),2),"hertz"


print "vm1coswt vm2sinwt squaring and adding gives ellipse"

from math import sqrt
voltag=1000##volt
#(1) velocity
vx=sqrt(2*1.6*10**-19*(voltag)/(9.11*10**-31))#
print "velocity x   =   %0.3e"%vx,"metre per second"
vox=1*10**5##metre per second intial velocity
vx=sqrt((vox)+((2*1.6*10**-19*voltag)/(2.01*1.66*10**-27)))#

print "velocity x   =   %0.2e"%vx,"metre per second"

from math import sqrt
voltag=2000##volt
d=15##centimetre
d1=3##centimetre
r1=((d**2+d1**2)/(6))*10**-2##centimetre to metre
vox=sqrt(2*1.6*10**-19*(voltag)/(9.11*10**-31))#
b=vox/((1.6*10**-19*r1)/(9.11*10**-31))#

print "transverse magnetic field   =   %0.2e"%b,"weber per metre square"

from math import sqrt
voltag=2000##volt
d=2*10**-2##metre
#(1) frequency
vx=sqrt(2*1.6*10**-19*(voltag)/(9.11*10**-31))#
durati=d/vx#
freque=1/(2*durati)#
print "max frequency  = %0.2e"%freque,"hertz"
#(2)
durati=60*durati#
print "duration electron between the plates   =   %0.2e"%durati,"second"#correction in book

from math import sqrt
voltag=800##volt


q=1.6*10**-19##coulomb
m=9.11*10**-31##kilogram
vox=sqrt(2*q*voltag/m)#

print "max velocity =  %0.2e"%vox,"metre per second"

from math import sqrt
voltag=2000##volt
d=1.5*10**-2##centimetre
d1=5*10**-3##metre
distan=50*10**-2##metre
#(1) velocity
vox=sqrt(2*1.6*10**-19*(voltag)/(9.11*10**-31))#
#(2) sensitivity
defsen=distan*d/(2*d1*voltag)#
#deflection factor
g=1/defsen#
print "velocity   =   %0.2e"%vox,"metre per second"
print "sensitivity   =   %0.2e"%defsen,"metre per volt"

print "deflection factor   =   ",round((g),2),"volt per metre"#correction in the book

from math import sqrt
voltag=2000##volt
d=50*10**-3##metre
#(1) velocity
vox=sqrt(2*1.6*10**-19*(voltag)/(9.11*10**-31))#
print "velocity   =   %0.2e"%vox,"metre per second"
#(2) fc
fc=vox/(4*d)#

print "fc   =   %0.2e"%fc,"hertz"

from math import asin, degrees
y=2.5##divisions
y1=1.25##divisions
y=y1/y#
w=degrees(asin(y))

print "phase angle   =   ",round((w),2),"degre"

