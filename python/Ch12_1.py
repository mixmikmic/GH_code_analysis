from math import sqrt
slope1=130
trivol=15##volt
d=0.5##watts
ig=sqrt(d/slope1)
vg=slope1*ig
r=(trivol-vg)/ig
print "source resistance   =   %0.2f"%((r)),"ohm"

from math import exp
latcur=50*10**-3##ampere
durpul=50*10**-6##second
induct=0.5##henry
r=20##ohm
voltag=100##volt
w=induct/r
inpcur=-(voltag/r)*((1)-exp(-durpul/w))
print "current   =   %0.3f"%(abs(inpcur)),"ampere"
print "input current less than required current"

latcur=4*10**-3##ampere
induct=0.1##henry
voltag=100##volt
durmin=induct*latcur/voltag
print "min duration   =   %0.2e"%((durmin)),"second"

from math import sqrt
slope1=3*10**3
egs=10##volt
d=0.012##watts
ig=sqrt(d/slope1)
vg=slope1*ig
r=(egs-vg)/ig

print "source resistance   =   %0.2f"%((r)),"ohm"##it is not given in the book

slope1=16
durmax=4*10**-6##second
curmin=500*10**-3##ampere
voltag=15##volt
#(1) resistance
vg=slope1*curmin
r=(voltag-vg)/curmin
#(2)
d=vg*curmin
freque=0.3/(d*durmax)
print "resistance   =   %0.2f"%((r)),"ohm"
print "frequency   =   %0.2f"%((freque)),"hertz"

c1=20*10**-12##farad
limcur=16*10**-3##ampere
w=(limcur/c1)*10**-6##convert second to microsecond
print "change of voltage   =   %0.2f"%((w)),"volt per microsecond"

from math import sqrt
ratcur=3000##ampere
freque=50##hertz
i=sqrt(ratcur**2/2)
print "current   =   %0.2f"%((i)),"ampere"
i=((ratcur)/sqrt(2))**2/(2*freque)
print "current   =   %0.2f"%((i)),"ampere square second"

from __future__ import division
from math import log
voltag=30##volt
w=0.51
i1=10*10**-6##ampere
v1=3.5##volt
curen1=10*10**-3##ampere
freque=60##hertz
tridun=50*10**-6##second
pinvol=w*voltag+0.6
r=(voltag-pinvol)/i1
print "max limit resistance   =   %0.2f"%((r)),"ohm"
r=(voltag-v1)/(curen1)
print "min limit resistance   =   %0.2f"%((r)),"ohm"
capac1=0.5*10**-6##farad
r=(1/freque)*(1/(capac1*log(1/(1-w))))
print "resistance   =   %0.2e"%((r)),"ohm"
rb2=10**4/(w*voltag)
rb1=tridun/capac1
print "rb1   =   %0.2f"%((rb1)),"ohm"
print "rb2   =   %0.2f"%((rb2)),"ohm"
print "peak voltage   =   %0.2f"%((pinvol)),"volt"

re=1*10**3##ohm
i1=5*10**-3##ampere

voltag=re*i1+2
print "voltage   =   %0.2f"%((voltag)),"volt"


print "this voltage makes to off"

