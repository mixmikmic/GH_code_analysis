alpha=0.98#
vbe=0.7##base emitter voltage volt
ie=-4*10**-3##emitter current
vc=12##colector voltage volt
colr=3.3*10**3##ohms
colCurrent=ie*(-alpha)#
baseCurrent=0.02*ie#
vbn=vbe+(-4*10**-3*100)#
i2=-vbn/(10*10**3)#
i1=-(baseCurrent+i2)#
vcn=(vc-((colCurrent+i1)*colr))#
v1=vcn-0.9#
r1=v1/i1#
print "r1   =   %0.2f"%(abs(r1)),"ohm"

colvoltag=12##volts
vbe=5##volts
colcur=10*10**-3##ampere
vce=5##volts
beta1=50#
ib=colcur/beta1#
rb=(vbe-0.7)/ib#
rc=(12-vbe)/colcur#
#when 100ohm included
print "rb   =   %0.2f"%(rb),"ohm"
print "rc   =   %0.2f"%(rc),"ohm"
rb=(vce-0.7-(colcur+ib)*beta1)/ib#

print "rb at emitter resistance 100ohm   =   %0.2f"%(rb),"ohm"#correction in the book

from math import log
#given
reveri=2*10**-6##ampere at 25
icb=2*10**-6*2**5##ampere at 75
basevoltag=5##volt
#(1)
rb=(-0.1+basevoltag)/(icb)#
print "max resistance   =   %0.2f"%((rb)),"ohm"#correction in the book
#(2)
basevoltag=1#
rb=100*10**3#
reveri=(-0.1+basevoltag)/rb#
q=reveri/(2*10**-6)#
w=q**10#
u=log(w)
t=25+(u/log((2)))#
print "baseresistance   =   %0.2f"%((rb)),"ohm"
print "temperature   =   %0.2f"%((t)),"celsius"

#given
vbe=0.8##volt
beta1=100#
vce=0.2##volt
rb=200*10**3##ohm
bascur=(6-vbe)/rb#
colres=(10-vce)/(beta1*bascur)#
print "min resistance   =   %0.2f"%((colres)),"ohm"

beta1=100#
colres=3*10**3##collector resistance #ohm
rb=8*10**3##ohm
r1=500##ohm
voltag=5##volt
#(1)
ib=(-voltag+0.7)/((1+beta1)*r1+(rb))#
ic=beta1*ib#
vce=(-10-ic*(colres)+r1*(ib+ic))#
vcb=vce+0.7#
#(2)
volmin=-0.2+abs(ib+ic)*r1#
re=-(0.7+rb*ib+voltag)/((1+(beta1))*ib)#
print "in saturation mode"
print "vo   =   %0.2f"%((volmin)),"volt"#correction in the book
print "emitter resistance   <   %0.2f"%((re)),"ohm"

vcc=12##volt
rb=12*10**3##ohm
colres=2*10**3##ohm
beta1=100#
vb=0.7##volt
vce=0.1##volt

for q in range(1,3):
    if q==1:
        vbb=1
    else:
        vbb=12
    
    ib=(vbb-vb)/rb
    ic=beta1*ib
    ie=ic+ib
    vce=vcc-ic*colres
    if q==2 :
        ic=(vcc-0.1)/colres
    

    print "the operating point at vbb   =   %0.2f"%((vbb)),"volt ic   =   %0.2e"%((ic)),"ampere vce   =   %0.2f"%((vce))," volt"

beta1=ic/ib#

print "beta at saturation   =   %0.2f"%((beta1))

vbe=0.65##volt
colres=2*10**3##ohm
voltag=10##volt
i1=voltag/10#
q=(1.65-vbe)/(1*10**3)#


print "current   =   %0.2e"%((q)),"ampere"

vcc=12##volt
r1=10*10**3##ohm
colres=1*10**3##ohm
re=5*10**3##ohm
rb=5*10**3##ohm
beta1=100#
vbe=0.7##volt
basvol=vcc*10/20#
ib=((basvol-vbe)/(rb+beta1*rb))#
ic=beta1*ib#
vce=vcc-ic*(colres+re)#
print "vce   =   %0.2f"%((vce)),"volt"
print "collector current   =   %0.2e"%((ic)),"ampere"

colres=330##ohm
re=0.1*10**3##ohm
vcc=12##volt
vce=0.2##volt
revcur=18*10**-3#ampere
ib=0.3*10**-3##ampere
stability=10#
beta1=100#
colres=0.330##ohm
re=0.1*10**3##ohm
vbe=0.2#
rb=(((1+beta1)*re)/10-((1+beta1)*re))/(1-10.1)#
vb=2+ib*rb#
w=vcc/vb#
q=w-1#
r1=1.2*10**3#
r=q*1.2*10**3#
print "r1   =   %0.2f"%((q)),"times r2"
print "if r2 is 1200ohm"
print "r1   =   %0.2f"%((r)),"ohm"

print "r2   =   %0.2f"%((r1)),"ohm"

alpha1=0.99#
ib=25*10**-6##ampere
icb=200*10**-9##ampere
beta1=alpha1/(1-alpha1)#
ic=beta1*ib+(beta1+1)*icb#
print "collector current   =   %0.2e"%((ic)),"ampere"
ie1=(ic-icb)/alpha1#
print "emitter current   =   %0.2e"%((ie1)),"ampere"
ic=beta1*ib#
print "collector current with ib   =   %0.2e"%((ic)),"ampere"
ie=ic/alpha1#
print "emitter current   =   %0.2e"%((ie)),"ampere"
w=(ie1-ie)/ie1#
print "error   =   %0.2e"%((w))

from __future__ import division
vcc=26##volt
colres=20*10**3##ohm
re=470##ohm
beta1=45#
vce=8##volt
ib=(vcc-vce)/((1+beta1)*(colres+re))#
ic=beta1*ib#
r1=((vcc-colres*(ib+ic)-re*(ib+ic)-(0.7)))/ib#
print "resistance   =   %0.2f"%((r1)),"ohm"
stability=(1+beta1)/(1+(beta1*re)/(re+colres))#
print "stability   =   %0.2f"%((stability))
#correction required in the book

vcc=1.5#volt in book should be changed as 1.5
colres=1.5*10**3##ohm
emresi=0.27*10**3##ohm
r1=2.7*10**3##ohm
r=2.7*10**3##ohm
beta1=45#
basre1=690##ohm
voltag=r*vcc/(r*r1)#
basres=(r*r1)/(r+r1)#
vbe=0.2#
for q in range (1,3):
    if q==2 :
        print "resistance   =   %0.2e"%((basre1)),"ohm"
        basres=basres+basre1
    
    bascur=(((voltag+vbe)))/(basres+(45*(emresi)))
    colcur=beta1*bascur
    vce=(vcc+colcur*colres+(bascur+colcur)*emresi)
    print "current   =   %0.2e"%((colcur)),"ampere"
    print "vce   =   %0.2f"%((vce)),"volt"

beta1=25#
colres=2.5*10**3##ohm
vcc=10##volt
vce=-5##volt
ic=-(vcc+vce)/colres#
ib=ic/beta1#
rb=vce/ib#
stability=(1+beta1)/((1+beta1)*((colres)/(colres+rb)))#
print "base resistance   =   %0.2f"%((rb)),"ohm"#correction in book
print "stability   =   %0.2f"%((stability))

therre=8##celsius per watts
tepera=27##celsius ambient temperature
potran=3##watt
tejunc=tepera+(therre*potran)#
print "junction temperature   =   %0.2f"%((tejunc)),"celsius"

from __future__ import division
ambtep=40##celsius
juntep=160##celsius
hs_a=8#
j_c=5#
c_a=85#
j_a=(j_c)+(c_a*hs_a)/(c_a+hs_a)#
podiss=(juntep-ambtep)/j_a#
print "dissipation   =   %0.2f"%((podiss)),"watt"

from __future__ import division
emicur=1*10**-3##ampere
colcur=0.995*10**-3##ampere
alpha1=colcur/emicur#
beta1=alpha1/(1-alpha1)#
print "alpha   =   %0.2f"%((alpha1))
print "beta   =   %0.2f"%((beta1))

from __future__ import division
beta1=100#
alpha1=beta1/(beta1+1)#

print "alpha   =   %0.2f"%((alpha1))

from __future__ import division
rb=200*10**3##ohm
rc=2*10**3##ohm
vcc=20##volt
ib=(vcc)/(rb+200*rc)#
ic=200*ib#
print "ic   =   %0.4f"%((ic)),"ampere"
#correction required in book

from __future__ import division
alpha1=0.98#
revcur=1*10**-6##ampere
emicur=1*10**-3##ampere
colcur=alpha1*emicur+revcur#
bascur=emicur-colcur#
print "collector current   =   %0.2e"%((colcur)),"ampere"
print "base current   =   %0.2e"%((bascur)),"ampere"

from __future__ import division
colcur=100*10**-3##ampere
ouresi=20##ohm
r=200##ohm
r1=100##ohm
vcc=15##volt
basvol=((r1)/(r+r1))*vcc#
em1res=basvol/colcur#
vce=vcc-(ouresi+em1res)*colcur#
print "vce   =   %0.2f"%((vce)),"volt"
print "emitter resistance   =   %0.2f"%((em1res)),"ohm"

from __future__ import division
colres=1*10**3##ohm
beta1=50#
vbe=0.3##volt
vcc=6##volt
rb=10*10**3##ohm
re=100##ohm
em1cur=((vcc-vbe)*(beta1+1))/((rb+((beta1+1)*re)))#
for q in range(1,3):
    if q==2 :
        colres=1*10**3#
        vce=vcc-(colres+re)*em1cur#
        ic=vcc/(colres+re)#
        print "collector to emitter   =   %0.2f"%((vce)),"volt"
        print "collector current   =   %0.3f"%((ic)),"ampere"
    
    if q==1 :
        colres=50#
        rb=100#
        vce=vcc-(colres+rb)*em1cur#
        print "emitter current   =   %0.3f"%((em1cur)),"ampere"
        print "collector to emitter   =   %0.3f"%((vce)),"volt"
    

from __future__ import division
beta1=99#
stability=5#
vbe=0.2##volt
colres=2.5*10**3##ohm
vce=6##volt
ven=5.5##volt
vcc=15##volt
vcn=vce+ven#
colvol=vcc-vcn##voltage across collector resistance
ic=colvol/colres#
ib=ic/beta1#
colre1=ven/ic#
rb=stability*colre1/(1-(stability/(1+beta1)))##correction in the book taken collector resistance as 3.13*10**3ohm but it is 3.93*10**3ohm
v1=(ib*rb)+(vbe)+((ib+ic)*colre1)#
r=rb*vcc/v1#
r1=r*v1/(vcc-v1)#
print "resistance   =   %0.2f"%((colre1)),"ohm"
print "resistance r1    =   %0.2f"%((r)),"ohm"
print "resistance r2   =   %0.2f"%((r1)),"ohm"

from __future__ import division
beta1=50#
vbb=5##volt
rb=10*10**3##ohm
colres=800##ohm
re=1.8*10**3##ohm
vcc=5##volt
ib=(0.7-vbb)/((rb)+(beta1+1)*re)##correction in book
re=beta1*ib#
ie=(ib+re)#
vce=vcc-colres*re-re*ie#
vcb=(vce-0.7)#
print "base current   =   %0.2e"%((ib)),"ampere"
print "collector current   =   %0.2e"%((re)),"ampere"
print "emitter current   =   %0.2e"%((ie)),"ampere"
print "vcb   =   %0.2f"%((vcb)),"volt"#correction in book
print "the collector base junction is reverse biased the transistor in active region"

from __future__ import division
r=40*10**3##ohm
r1=5*10**3##ohm
colres=r1#
beta1=50#
em1res=1*10**3##ohm
vcc=12##volt
rth=r*r1/(r+r1)#
v1=r1*vcc/(r1+r)#
bascur=(v1-0.3)/(rth+(beta1*em1res))#
colcur=beta1*bascur#
vce=vcc-(colres+em1res)*colcur#
print "collector current   =   %0.2e"%((colcur)),"ampere"
print "collector emitter voltage   =   %0.2f"%((vce)),"volt"

from __future__ import division
colcur=8*10**-3##ampere
re=500##ohm
vce=3##volt
beta1=80#
vcc=9##volt
ib=colcur/beta1#
rb=(vcc-(1+beta1)*(ib*re))/ib#
print " base resistance   =   %0.f"%((rb)),"ohm"

from __future__ import division
vcc=10##volt
basres=1*10**6##ohm
colres=2*10**3##ohm
em1res=1*10**3##ohm
beta1=100#
bascur=vcc/(basres+(beta1+1)*(em1res))#
colcur=beta1*bascur#
em1cur=colcur+bascur#
print "base current   =   %0.2e"%((bascur)),"ampere"
print "collector current   =   %0.2e"%((colcur)),"ampere"#correction in book
print "emitter current   =   %0.2e"%((em1cur)),"ampere"#correction in book

from __future__ import division
alpha1=0.99#
rebacu=1*10**-11##ampere
colres=2*10**3##ohm
vcc=10##volt
bascur=20*10**-6##ampere
beta1=alpha1/(1-alpha1)#
i1=(1+beta1)*rebacu#
colcur=beta1*bascur+i1#
em1cur=-(bascur+colcur)#
vcb=vcc-colcur*colres#
vce=vcb-0.7#
print "collector current   =   %0.2e"%((colcur)),"ampere"
print "emitter current   =   %0.2e"%((em1cur)),"ampere"
print "collector emitter voltage   =   %0.2f"%((vce)),"volt"

from __future__ import division
beta1=100#
revcur=20*10**-9##ampere
colres=3*10**3##ohm
rb=200*10**3##ohm
vbb=5##volt
vcc=11##volt
em1res=2*10**3##ohm
ib=(vbb-0.7)/rb#
ic=beta1*ib#
ie=ib+ic#
print "base current   =   %0.2e"%((ib)),"ampere"
print "collector current   =   %0.2e"%((ic)),"ampere"
print "emitter current   =   %0.2e"%((ie)),"ampere"#question asked only currents
#2*10**3 ohm added to emitter
ib=-(0.7-vcc)/(rb+((1+beta1)*em1res))#
ic=beta1*ib#
ie=ib+ic#
print "base current   =   %0.2e"%((ib)),"ampere"#correction in book
print "collector current   =   %0.2e"%((ic)),"ampere"
print "emitter current   =   %0.2e"%((ie)),"ampere"#question asked only currents

from __future__ import division
em1cur=2*10**-3##ampere
v1=12##volt
vcc=12##volt
format(12)#
colres=5*10**3##ohm
em1res=v1/em1cur#
colcur=em1cur#
voltag=colcur*colres##ic*r
v1=vcc-(colres*colcur)#
print "emitter current   =   %0.2e"%((em1cur)),"ampere"
print "collector current   =   %0.2e"%((colcur)),"ampere"
print "voltage   =   %0.2f"%((voltag)),"volt"
print "vcb   =   %0.2f"%(abs(v1)),"volt"
print "emitter resistance   =   %0.2f"%((em1res)),"ohm"

from __future__ import division
vbb=4##volt
ib=50*10**-6##ampere
for q in [0, 0.7, 4, 12]:
    if q==0 :
        rb=(vbb-q)/ib#
        print "resistance at %0.1f"%((q)),"volt   %0.2f"%((rb)),"ohm"
    elif q==0.7:
        rb=(vbb-q)/ib
        print "resistance at %0.2f"%((q)),"volt   %0.2f"%((rb)),"ohm"
    elif q==4:
        print "vbb at 12volt"
        q=0
        vbb=12
        rb=(vbb-q)/ib
        print "resistance at %0.2f"%((q)),"volt   %0.2f"%((rb)),"ohm"
    else:
        q=0.7#
        vbb=12#
        rb=(vbb-q)/ib#
        
        
        print "resistance at %0.2f"%((q)),"volt   %0.2f"%((rb)),"ohm"

from __future__ import division
ic=5.2*10**-3##ampere
ib=50*10**-6##ampere
icb=2*10**-6##ampere
beta1=(ic-icb)/(ib+icb)#
print "beta   =   %0.2f"%((beta1))
ie=ib+ic#

print "ie   =   %0.3f"%((ie)),"ampere"
alpha1=(ic-icb)/ic#
print "alpha   =   %0.2f"%((alpha1))



ic=10*10**-3##ampere
ib=(ic-(beta1+1)*(icb))/beta1#


print "ib   =   %0.2e"%((ib)),"ampere"
#correction required in the book

from __future__ import division
beta1=160
vb=-0.8##volt
re=2.5*10**3##ohm
vcc=10##volt
for q in [160, 80]:
    ib=(vcc-vb)*10**2/((re)*(1+q)*400)#
    ic=q*ib#
    colres=1.5*10**3##ohm
    print "collector current at beta %0.2f"%((q)),"   =   %0.2e"%((ic)),"ampere"
    #correction required in the book
    ie=(1+beta1)*ib#
    vce=-(vcc-colres*ic-re*ie)#
    print "vce at beta %0.2f"%((q)),"   =   %0.2f"%((vce)),"volt"
    #correction required in the book

from __future__ import division
vb=0.7##volt
vce=7##volt
ic=1*10**-3##ampere
vcc=12##volt
beta1=100#
colres=(vcc-vce)/ic#
ib=ic/beta1#
#rb
rb=(vcc-vb-ic*colres)/ib#
print "rb   =   %0.2f"%((rb))," ohm"
#stability
stability=(1+beta1)/(1+beta1*(colres/(colres+rb)))#
print "stability   =   %0.2f"%((stability))
#beta=50
beta1=50#
print "new point"
ib=(vcc-vb)/(beta1*colres+rb)#
ic=beta1*ib#
print "ic   =   %0.2e"%((ic))," ampere"
vce=vcc-(ic*colres)#
print "vce   =   %0.2f"%((vce))," volt"

from __future__ import division
vcc=16##volt
colres=3*10**3##ohm
re=2*10**3##ohm
r1=56*10**3##ohm
r2=20*10**3##ohm
alpha1=0.985#
vb=0.3##volt
#coordinates
beta1=alpha1/(1-alpha1)#
v1=vcc*r2/(r1+r2)#
rb=r2/(r1+r2)#
ic=(v1-vb)/((rb/beta1)+(re/beta1)+re)#
print "new point"
print "vce   =   %0.2f"%((v1))," volt"
print "ic   =   %0.2e"%((ic))," ampere"

from __future__ import division
vce=12##volt
ic=2*10**-3##ampere
vcc=24##volt
vb=0.7##volt
beta1=50#
colres=4.7*10**3##ohm
#re
re=((vcc-vce)/(ic))-colres#
print "re   =   %0.2f"%((re))," ohm"
#r1
ib=ic/beta1#
v1=ib*3.25*10**3+vb+(ib+1.5*10**3)#
r1=3.25*18*10**3/2.23#
print "r1   =   %0.2f"%((r1))," ohm"
#r2
r2=26.23*2.23*10**3/(18-2.3)#
print "r2   =   %0.2f"%((r2))," ohm"

from __future__ import division
colres=3*10**3##ohm
rb=150*10**3##ohm
beta1=125#
vcc=10##volt
v1=5##volt
vb=0.7##volt
ib=(v1-vb)/rb#
print "ib   =   %0.2e"%((ib))," ampere"
ic=beta1*ib#
ie=ic+ib#
print "ic   =   %0.2e"%((ic))," ampere"
print "ie   =   %0.2e"%((ie))," ampere"#correction in the book in question to find only currents

from __future__ import division
beta1=50#
vb=0.6##volt
vcc=18##volt
colres=4.3*10**3##ohm
ic=1.5*10**-3##ampere
vce=10##volt
stability=4#
r1=(vcc-vce)/ic#
re=r1-colres#
w=(beta1+1)*(stability)*re/(1+beta1-stability)#
print "re   =   %0.2f"%((re)),"ohm"
print "rb   =   %0.2f"%((w)),"ohm"#correction in the book

from __future__ import division
re=100##ohm
beta1=100#
rb=1*10**3##ohm
stability=(1+beta1)/(1+beta1*(re/(re+rb)))#
r1=3.8#r2
print "r1   =   3.8*r2"#correction in the book not given in question

from __future__ import division
from math import log10
icb=2*10**-6##ampere
vbb=1##volt
r1=50*10**3##ohm
#current increases every 10celsius rb at 75celsius
vb=-0.1##volt
icb=2**6*10**-6##at 75celsius
rb=(vb+vbb)/icb#
print "rb at 75 celsius   =   %0.2f"%((rb)),"ohm"
icb=(vb+vbb)/r1#
print "icb   =   %0.2e"%((icb)),"ampere"
w=(log10(icb*10**6)*20/log10(2))-25#
print "temperature at which current till max   =   %0.2f"%((w)),"celsius"

from __future__ import division
vb=0.8##volt
beta1=100#
vce=0.2##volt
vcc=10##volt
rb=200*10**3##ohm
#collector resistance
ib=(5-0.7)/rb#
colres=(vcc-vce)/(beta1*ib)#
print "min collector resistance   =   %0.2f"%((colres)),"ohm"

from __future__ import division
alpha1=0.98#
alph11=0.96#
vcc=24##volt
colres=120##ohm
ie=100*10**-3##ampere
beta1=alpha1/(1-alpha1)#
bet11=alph11/(1-alph11)#
ib2=ie/(1+bet11)#
ie1=-ib2#
print "ib2   =   %0.2e"%((ib2)),"ampere"
print "ie1   =   %0.2e"%((ie1)),"ampere"


ic2=bet11*ib2#
ib1=ib2/(1+beta1)#
ic1=beta1*ib1#
print "ic2   =   %0.2e"%((ic2)),"ampere"
print "ib1   =   %0.2e"%((ib1)),"ampere"
print "ic1   =   %0.2e"%((ic1)),"ampere"
ic=ic1+ic2#
vce=vcc-ic*colres#
ib=ib1#
w=ic/ib#
q=-ic/ie#
print "ic   =   %0.2e"%((ic)),"ampere"
print "ic/ib   =   %0.2f"%((w))
print "ic/ie   =   %0.2f"%((q))
#correction required in the book
print "vce   =   %0.2f"%((vce)),"volt"

