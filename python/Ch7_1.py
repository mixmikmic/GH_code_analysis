from sympy import symbols, solve
rd=12*10**3##ohm
r=1*10**6##ohm
resour=470##ohm
vdd=30##volt
idss=3*10**-3##ampere
vd=2.4##volt
v = symbols('v')
vgs=[0.24, 2.175, 1.41]
expr = vgs[0]*v**2+vgs[1]*v+vgs[2]
vgs=-solve(expr,v)[1]
vgs=0.7
id=idss*((1-(vgs/vd)))**2
vds=vdd-id*(rd+resour)
g=(2*idss/vd)*(1-((vgs/vd)))
volgai=-g*rd
print "vgs   =   %0.2f"%((vgs)),"volt"
print "id   =   %0.2e"%((id)),"ampere"
print "vds   =   %0.2f"%((vds)),"volt"
print "voltage gain   =   %0.2f"%((volgai))

from __future__ import division
idss=1*10**-3##ampere
pinvol=1##volt
q=10##volt
rd=56*10**3##ohm
vdd=24##volt
dracur=(vdd-q)/rd
vgs=0.5
r1=vgs/dracur
print "r1   =   %0.2f"%((r1)),"ohm"

ids=4*10**-3##ampere
vp=4##volt
r=1.3*10**3#ohm
r1=200*10**3##ohm
vdd=60##volt
drares=18*10**3##ohm
soresi=4*10**3##ohm
rth=(r*r1)/(r+r1)
vth=r1*(1-vdd)/(1500*10**3)
id=-2.25*10**-3
vds=-vdd-(drares+soresi)*id
print "id   =   %0.2e"%(abs(id)),"ampere"
print "vds   =   %0.2f"%(abs(vds)),"volt"

from math import sqrt
idss=10*10**-3##ampere
pinvol=-1##volt
ids=6.4*10**-3##ampere
vgs=-(sqrt(ids/idss)-(1))*pinvol
r=pinvol/ids
print "source resistance   =   %0.2f"%(abs(r)),"ohm"

from math import log
v1=2##volt
vgs=4##volt
voltag=5##volt
q=5*10**-3##ampere per volt square
id=q*(vgs-v1)
durati=10**-7*log(4)

print "duration   =   %0.2e"%((durati)),"second"

idss=1*10**-3##ampere
pinvol=-5##volt
tracon=(2*idss)/abs(pinvol)
print "max transconductance   =   %0.2e"%((tracon)),"mho"

from math import sqrt
vdd=10##volt
beta1=10**-4##ampere per square volt
ids=0.5*10**-3##ampere
voltag=1##volt
vgs=(sqrt(ids/beta1)+(1))
rd=(vdd-vgs)/ids

print "vgs   =   %0.2f"%((vgs)),"volt"
print "rd   =   %0.2f"%((rd)),"ohm"

v1=2##volt
ids=4*10**-3##ampere

rd=910##ohm
r1=3*10**3##ohm
r=12*1**6##ohm
r11=8.57*10**6##ohm
vdd=24##volt
vg=vdd*(r11/(r+(r11)))
id=3.39*10**-3
vgsq=vg-id*r1
vdsq=vdd-id*(rd+r1)
vdgq=vdsq-vgsq
print "point %0.2f"%(vdsq),">%0.2f"%(v1),"volt"
print "vds greater than 2volt the point in pinch"

