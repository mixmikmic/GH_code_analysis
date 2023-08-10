ic=1*10**-3##ampere
vcc=5##volt
colres=2*10**3##ohm
r1=1.4*10**3##ohm
re=100##ohm
beta1=100
rb=100##ohm
v1=0.026
c1=25*10**-6##farad
g1=ic/v1
freque=10*10**3##hertz
xc=1/(2*freque*3.14*c1)
volgai=-beta1*colres/(r1+0.1*10**3+2.5*10**3)
print "voltage gain   =   %0.2f"%((volgai))
ri=(0.1+2.5)*10**3-((xc.imag)*(1+beta1))
print "input resistance   =   %0.2f"%((ri)),"ohm"
#ce removed
volgai=-beta1*colres/((r1+0.1*10**3+2.5*10**3)+(101/1000)*10**3*100)
print "ce removed"
print "voltage gain   =   %0.2f"%((volgai))
ri=(0.1+2.5)*10**3+100*101/1000*10**3
print "input resistance   =   %0.2f"%((ri)),"ohm"

ic=1.3*10**-3##ampere
colres=2*10**3##ohm
re=500##ohm
v1=0.026##volt
beta1=100
vcc=15##volt
c1=10*10**-6##farad
ib=ic/beta1
ri=0.01/ib
volgai=beta1*colres*ib/0.01
print "voltage gain   =   %0.2f"%((volgai)),"<180"
print "voltage gain reduced ce removed"
print "when cb is   short circuited the voltage gain increased"

colres=4*10**3##ohm
r1=4*10**3##ohm

rb=20*10**3##ohm
r=1*10**3##ohm
hie=1.1*10**3##ohm

#current gain
ri=rb*hie/(rb+hie)
curgai=(1/2.04)*(rb/(rb+(hie)))*(-50*colres/(colres+(r1)))
print "current gain   =   %0.2f"%((curgai))
#voltage gain
volgai=curgai*r1/r
print "voltage gain   =   %0.2f"%((volgai))
#transconductance
conduc=volgai/r1
print "transconductance   =   %0.2f"%((conduc)),"ampere per volt"
#transresistance
resist=volgai*r
print "transresistance   =   %0.2f"%((resist)),"ohm"
#input resistance
print "input resistance   =   %0.2f"%((ri)),"ohm"
#output resistance
resist=40*10**3*colres/(40*10**3+colres)



print "output resistance   =   %0.2f"%((resist)),"ohm"

ib=20*10**-6##ampere
beta1=500
re=10##ohm correction in the book
r1=4.7*10**2##ohm correction in the book
ic=ib*beta1
voltag=ic*r1##voltage drop at 4.7*10**3ohm
vc=(10-voltag)
rb=(vc-0.6)/ib
print "rb   =   %0.2f"%((rb)),"ohm"
#re included
voltag=ic*re##voltage drop at re
vb=(0.6+voltag)
rb=(vc-vb)/ib
print "rb including emitter resistance   =   %0.2f"%((rb)),"ohm"

from __future__ import division
from math import log10
av=12480
fedbac=8##decibel
volgai=20*log10(av)##gain without fedback
volga1=volgai-fedbac
beta1=((av/5000)-1)/av

print "voltage gain with fedback   =   %0.2f"%((volga1)),"decibel"
print "beta   =   %0.2e"%((beta1))

beta1=100
r1=1.5*10**3##ohm
vcc=10##volt
r=100*10**3##ohm
vb=((vcc)/(r+10*10**3))*10*10**3
ie=0.3/100
ib=ie/beta1
print "collector current   =   %0.2e"%((ie)),"ampere"
print "emitter current   =   %0.2e"%((ie)),"ampere"
print "base current   =   %0.2e"%((ib)),"ampere"

hie=800##ohm
he=50*10**-6##mho
hfe=-55
z1=2*10**3##ohm
curgai=hfe/(1+he*z1)
zi=hie
volgai=curgai*z1/zi
powgai=volgai*curgai
#if hoe neglected
av=137.5
hfe=-55
w=((av-abs(volgai))*100)/abs(volgai)
ap=hfe*(-av)
w1=((ap-powgai)*100)/powgai
print "voltage gain   =   %0.2f"%((volgai))


print "power gain   =   %0.2f"%((powgai))
print "error without hoe   =   %0.2f"%((w))
print "error   =   %0.2f"%((w1))

rb=5*10**3##ohm
vcc=20##volt
r=10*10**3##ohm
colres=5*10**3##ohm
vb=vcc*r/(r+r)
beta1=50
v1=0.6##volt
ib=(vb-v1)/(1+beta1*colres)
ic=beta1*ib
vc=vcc-ic*1*10**3
vce=vc-rb*(ic+ib)
print "emitter current   =   %0.2e"%((ic+ib)),"ampere"
print "vc   =   %0.2f"%((vc)),"volt"
print "collector emitter voltage   =   %0.2f"%((vce)),"volt"

hib=25##ohm
hfb=0.999
hob=10**-6##ohm
colres=10*10**3##ohm
#voltage gain
curgai=hfb/(1+hob*colres)
zi=hib+hob*colres*curgai
volgai=curgai*colres/(zi)
print "voltage gain   =   %0.2f"%((volgai))
#correction required in the book

re=1*10**3##ohm
hie=100##ohm
hfe=100
#voltage gain
volgai=1/((1+(hie/(2*(1+hfe)*re))))
#ri
ri=(hie/2)+(1+hfe)*re
print "voltage gain   =   %0.2f"%((volgai))
print "input resistance   =   %0.2f"%((ri)),"ohm"

beta1=90
re=2*10**3##ohm
rb=240*10**3##ohm
vcc=20
ib=(vcc-0.7)/(rb+(1+beta1)*(re))
ic=beta1*ib
vce=vcc-(ib+ic)*re
print "emitter current   =   %0.2e"%((ib+ic)),"ampere"
print "vce   =   %0.2f"%((vce)),"volt"

hfe=110
hie=1.6*10**3##ohm
hoe=20*10**-6##ohm
colres=4.7*10**3##ohm
hre=2*10**-4
r1=470*10**3##ohm
curgai=-hfe/(1+hoe*colres)
ri=hie+hre*curgai*colres
volgai=curgai*colres/ri
y1=hoe-((hfe*hre)/(hie+1*10**3))
z1=1/y1
print "voltage gain   =   %0.2f"%((volgai))
print "current gain   =   %0.2f"%((curgai))
print "impedance   =   %0.2f"%((z1)),"ohm"
r0=z1*colres/(z1+colres)
curgai=-hfe
ri=hie
print "parameters using approxmiate"
volgai=curgai*(colres)/ri
print "voltage gain   =   %0.2f"%((volgai))
#correction required in the book
print "current gain   =   %0.2f"%((curgai))
print "impedance   =   %0.2f"%((z1)),"ohm"

from __future__ import division
re=1*10**3##ohm
hie=1000##ohm
hfe=99
#inptut resistance
ri=hie+((1+hfe)*(hie+1+hfe*re))


print "input resistance   =   %0.2e"%((ri)),"ohm"##correction in the book
#voltage gain
volgai=((1+hfe)*(1+hfe)*re)/ri
print "voltage gain   =   %0.2f"%((volgai))


#current gain
curgai=-((1+hfe)*(1+hfe))


print "current gain   =   %0.2f"%((curgai))

hie=2*10**3##ohm
beta1=100
colres=5*10**3##ohm
volgai=beta1*colres/hie
print "voltage gain   =   %0.2f"%((volgai)),"<180"
print "input impedance   =   %0.2f"%((hie)),"ohm"
print "current gain   =   %0.2f"%((beta1))

colres=4.7*10**3##ohm
beta1=150
r1=12*10**3##ohm
vcc=15##volt
re=1.2*10**3##ohm
rac=colres*r1/(colres+r1)
r=2*10**3##ohm
#voltage gain
volgai=beta1*rac/r
print "voltage gain   =   %0.2f"%((volgai))
r1=75*10**3##ohm
r2=7.5*10**3##ohm
#input impedance
zin=(r1*r2)/(r1+r2)
zin=zin*r/(zin+r)
print "input impedance   =   %0.2f"%((zin))
#coordinates
vb=vcc*r2/(r1+r2)
ie=vb/re
vce=vcc-((colres+re)*(ie))
print "coordinates ic   =   %0.2e"%((ie)),"ampere vce   =  %0.2f"%((vce)),"volt"

r1=2000##ohm
r=900##ohm
hie=1200##ohm
hre=2*10**-4
hfe=60
hoe=25*10**-6##ampere per volt
curgai=(hfe)/(1+hoe*r1)
print "current gain   =   %0.2f"%((curgai))
ri=hie+(curgai*r1)
print "input impedance   =   %0.2f"%((ri)),"ohm"
volgai=curgai*r1/ri
print "voltage gain   =   %0.2f"%((volgai))
admita=1/ri
admita=hoe-(-hfe*hre)/(hie+r)
r=1/admita
print "output resistance   =   %0.2f"%((r)),"ohm"

hfe=60
hie=500##ohm
ic=3*10**-3##ampere
zi=hie
rb=220*10**3##ohm
colres=5.1*10**3##ohm
z=colres
volgai=-hfe*colres/hie
curgai=-hfe
vcc=12##volt
ib=(vcc-0.6)/rb
ie=hfe*ib
re=0.026/ie
zi=hfe*re
z=colres
volgai=-colres/re
curgai=-hfe
print "voltage gain   =   %0.2f"%((volgai))
print "current gain   =   %0.2f"%((curgai))
print "input impedance   =   %0.2f"%((zi)),"ohm"
print "output impedance   =   %0.2f"%((z)),"ohm"

hie=3.2*10**3##ohm
hfe=100
r=40*10**3##ohm
r1=4.7*10**3##ohm
colres=4*10**3##ohm
rb=r*r1/(r+r1)
zi=hie*rb/(hie+rb)
z=colres
re=1.2*10**3##ohm
volgai=-hfe*colres/hie
print "input impedance   =   %0.2f"%((zi)),"ohm"
print "output impedance   =   %0.2f"%((z)),"ohm"
print "voltage gain   =   %0.2f"%((volgai))
curgai=-hfe*rb/(rb+hie)
print "current gain   =   %0.2f"%((curgai))
hie=833
#(1) load open
vi=1
ib=vi/hie
volgai=hfe*ib*1.5*10**3
#load closed
hoe=50
r2=2*10**3##ohm
ib=vi/(r2+hie)
vb=1.682
ib=(vb-0.6)/(rb+(1+hfe)*(re))
ic=hfe*ib
ie=ic+ib
re=0.026/ie
zi=rb*hfe*re/((rb)+(hfe*re))
print "parameters in re"
print "input impedance   =   %0.2f"%((zi)),"ohm"
z=colres
print "output impedance   =   %0.2f"%((z)),"ohm"
volgai=colres/(-re)
print "voltage gain   =   %0.2f"%((volgai))
curgai=-hfe*rb/(rb+hfe*re)
print "current gain   =   %0.2f"%((curgai))

from __future__ import division
hfe=120
hie=0.02##ohm
r1=5.8*10**3##ohm
r=27*10**3##ohm
colres=1.5*10**3##ohm
re=330*10**3##ohm
vcc=10##volt
vb=vcc*r1/(r1+r)
rb=(r*r1)/(r+r1)
ib=(vb-0.7)/(rb+((1+hfe)*re))
volgai=-hfe*ib*2*10**3
print "voltage gain   =   %0.3f"%((volgai))
#correction required in the book

from __future__ import division
freque=6*10**6##hertz
hfe=50
r1=500##ohm
g=0.04
rbb=100##ohm


c1=10*10**-12##farad
r=1000##ohm
rbe=hfe/g
ce=g/(2*3.14*freque)
c1=ce+c1*(1+g*r)
hie=rbb+rbe
resist=(r1+rbb)*rbe/(r1+rbb+rbe)
frequ2=1/(2*3.14*resist*c1)
curgai=-hfe*r1/(r1+hie)
volgai=(-hfe*r)/(r1+hie)
q=volgai*frequ2
print "upper frequency voltage gain   =   %0.2e"%(abs(q)),"hertz"##correction in the book
q=curgai*frequ2
print "upper current gain   =   %0.2e"%(abs(q)),"hertz"

from __future__ import division
hie=1*10**3##ohm
hre=2*10**-4
hoe=25*10**-6##ampere per volt
hfe=50
colres=1*10**3##ohm
curgai=-hfe/(1+hoe*colres)
print "current gain   =   %0.2f"%((curgai))
ri=hie-hfe*hre/(hoe+1/colres)
print "input resistance   =   %0.2f"%((ri)),"ohm"
volgai=curgai*colres/ri
print "voltage gain   =   %0.2f"%((volgai))
y1=hoe-((hfe*hre)/(hie+800))
r1=1/y1
print "output resistance   =   %0.2f"%((r1)),"ohm"
#approximate
print "approximate"
curgai=-hfe
print "current gain   =   %0.2f"%((curgai))
ri=hie
print "input resistance   =   %0.2f"%((ri)),"ohm"
volgai=-hfe*colres/hie
print "voltage gain   =   %0.2f"%((volgai))

from __future__ import division
rb1=7.5*10**3##ohm
rb2=6.8*10**3##ohm

rb3=3.3*10**3##ohm
re=1.3*10**3##ohm
colres=2.2*10**3##ohm
beta1=120
vcc=18##volt
vb1=rb3*vcc/(rb3+rb2+rb1)
ie1=(vb1-0.7)/(re)
re1=0.026/ie1
re2=0.026/ie1
volgai=colres/re2
print "voltage gain   =   %0.2f"%((volgai))

from __future__ import division
vcc=5##volt
colres=250##ohm
v1=5##volt
rb=25*10**3##ohm
beta1=200
vbs=0.8##volt
vcon=0.3##volt
icon=(vcc-vcon)/colres
ibon=icon/beta1
ibs=(v1-vbs)/rb
ic=(vcc-0.2)/colres
beta1=ic/ibs
print "forced beta   =   %0.2f"%((beta1))

from __future__ import division
vb=0.6##volt
beta1=100
ic=1*10**-3##ampere
vce=2.5##volt
re=300##ohm
vcc=5##volt
ib=ic/beta1
ie=ic+ib
ve=ie*re
vce=vce+ve
r3=(vcc-vce)/ic
vb=ve+vb
r1=(vcc-vb)/(vb/(10*10**3)+(ib))
print "resistance r1   =   %0.2f"%((r1)),"ohm"
print "resistance r3   =   %0.2f"%((r3)),"ohm"

from __future__ import division
vce2=7.5##volt
vb=0.7##volt
beta1=200
v1=25##volt
r1=10*10**3##ohm
vcc=15##volt
i1=(vcc-vb)/r1
r=(vcc-vce2)/i1
z1=beta1*v1/i1
z=v1/i1
print "input impedance q1   =   %0.2e"%((z)),"ohm"##correction in the book
print "input impedance q2   =   %0.2e"%((z1)),"ohm"

from __future__ import division
beta1=99
r1=1*10**3##ohm
g=beta1/r1
r=r1*((r1+r1)/(100))/((r1+((r1+r1)/(100))))
print "make input   =   0"
print "ground dc"
print "output resistance   =   %0.2f"%((r)),"ohm"

from __future__ import division
ic=0.5*10**-3##ampere
rb=100*10**3##ohm
v1=0.026##volt
r1=50##ohm
colres=1*10**3##ohm
g=ic/v1
volgai=g*colres
print "output resistance   =   %0.2f"%((colres)),"ohm"
print "input resistance very low"##not given in the book
print "voltage gain   =   %0.2f"%((volgai))

from __future__ import division
re=4*10**3##ohm
r1=4*10**3##ohm
hie=1.1*10**3##ohm
resist=10*10**3##ohm
hfe=50
rb=10*10**3##ohm
r=1*10**3##ohm
colres=5*10**3##ohm
#(1) current gain
ri=rb*hie/(rb+hie)
curgai=(1/2.04)*((rb)/(rb+hie))*((-hfe*colres)/(colres+r1))
print "current gain   =   %0.2f"%((curgai))
#(2) voltage gain
volgai=curgai*r1/r
print "voltage gain   =   %0.2f"%((volgai))
#(3) tranconductance
conduc=volgai/r1
print "transconductance   =   %0.2f"%((conduc)),"ampere per volt"
#transresistance
resist=resist*volgai
print "transresistance   =   %0.2f"%((resist)),"ohm"
print "input resistance   =   %0.2f"%((ri)),"ohm"
r=(40*10**3*colres)/(40*10**3+colres)
print "output resistance   =   %0.2f"%((r)),"ohm"

from __future__ import division
beta1=500
ib=20*10**-6##ampere
re=100##ohm
ic=beta1*ib
vc=ic*0.47*10**3##voltage drop across collector resistance
v1=(10-vc)
vb=v1-0.6
rb=vc/ib
print "base resistance   =   %0.2f"%((rb)),"ohm"
ve=re*ic
print "base resistance with re"
b=0.6+0.1
rb=(v1-b)/ib
print "base resistance   =   %0.2f"%((rb)),"ohm"

from __future__ import division
beta1=100
re=100##ohm
vcc=10##volt
colres=1.5*10**3##ohm
r=100*10**3##ohm
r1=10*10**3##ohm
vb=vcc*r1/(r1+r)
ie=0.3/re
ib=ie/beta1
print "collector current   =   %0.2e"%((ie)),"ampere"
print "base current   =   %0.2e"%((ib)),"ampere"
print "emitter current   =   %0.2e"%((ie)),"ampere"

