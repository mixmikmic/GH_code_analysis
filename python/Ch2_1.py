q=0.01##centimetre
sigma1=1##ohm centimetre inverse
q1=0.01##centimetre
sigm11=0.01##ohm centimetre inverse
iratio=(0.0224**2*2.11*20)*3.6**2/((3.11*(4.3**2*10**-6)**2*2.6*20*10**3))#
for q in range(0,2):
    if q==1:
        un=3800#
        up=1500#
        q=1.6*10**-19#
        ni=2.5*10#
    else:
        q=1.6*10**-19#
        up=500
        un=1300#
        ni=1.5*10

    
    b=un/up#
    sigmai=(un+up)*q*ni#

print "ratio of reverse saturation current   =   %0.2f"%((iratio))
##correction required in the book

sigma1=0.01##ohm centimetre inverse
area11=4*10**-3##metre square
q=0.01*10**-2##metre
un=1300.0#
up=500.0#
ni=1.5*10**15##per cubic centimetre
sigma1=(un+up)*1.6*10**-19*ni#
iratio=(4*10**-10*0.026*sigma1**2*2.6*2/10**-4)/3.6**2#
print "reverse current ratio   =   %0.2e"%((iratio))
##correction required in the book

a=4*10**-4##metre square
sigmap=1#
sigman=0.1#
de=0.15#
vtem=26*10**-3#
i=(a*vtem*((2.11)*(0.224))/((3.22)**(2)))*((1/de*sigman)+(1/de*sigmap))#
print "reverse saturation current   =   %0.2e"%(i),"ampere"#correction in the book

from math import log, exp
w=0.9#
voltaf=0.05##volt
revcur=10*10**-6##ampere
#(1) voltage
volrev=0.026*(log((-w+1)))##voltage at which the reverse saturation current at saturate
resacu=((exp(voltaf/0.026)-1)/((exp(-voltaf/0.026)-1)))##reverse saturation current
print "voltage at which the reverse saturation current at saturate   =   %0.2f"%((volrev)),"volt"
print "reverse saturation current   =   %0.2f"%((resacu)),"ampere"
u=0.1#
for q in range(0,3):
        reverc=revcur*(exp((u/0.026))-1)
        print "reverse saturation current %0.2f"%((u)),"   =   %0.3f"%((reverc)),"ampere"
        u=u+0.1#

a=1*10**-6##metre square
w=2*10**-6##thick centimetre
re=16#
eo=8.854*10**-12#
c=(eo*re*a)/w#
print "capacitance   =   %0.2e"%(c),"farad"

from math import sqrt
volbar=0.2##barrier voltage for germanium volt
na=3*10**20##atoms per metre
#(1) width of depletion layer at 10 and 0.1 volt

for q in [-10, -0.1, 0.1]:
    w=2.42*10**-6*sqrt((0.2-(q)))#
    print "width of depletion layer at %0.2f"%((q)),"   =   %0.2e"%((w)),"metre"#for -0.1volt correction in the book

#(d) capacitance
for q in [-10, -0.1]:
    capaci=0.05*10**-9/sqrt(0.2-q)#
    print "capacitance at %0.2f"%((q)),"   =   %0.2e"%((capaci)),"farad"

p=2##watts
voltaf=900*10**-3##volt
i1=p/voltaf#
r1=voltaf/i1#
print "maximum forward current   =   %0.2f"%(i1),"ampere"


print "forward diode resistance   =   %0.2f"%(r1),"ohm"

from math import atan, degrees
r=250##ohm
c=40*10**-6##farad
alpha1=180-degrees(atan(377*r*c))
print "alpha   =   %0.2f"%(alpha1),"degree"       

from math import sqrt
i1=0.1##current in ampere
vms=40##rms voltage in volts
c=40*10**-6##capacitance in farad
r1=50##resistance in ohms
ripple=0.0001#
induct=((1.76/c)*sqrt(0.472/ripple))##inductance
outv=(2*sqrt(2)*vms)/3.14-i1*r1##output voltage
print "inductance   =   %0.2f"%(induct),"henry"#correction in the book
print "output voltage   =   %0.2f"%(outv),"volt"

from math import sqrt
voltag=40##volt
i1=0.2##ampere
c1=40*10**-6##farad
c2=c1#
induct=2##henry
#(1) ripple
vdc=2*sqrt(2)*voltag/3.14#
r1=vdc/i1#
induc1=r1/1130#
v1=voltag/(3*3.14**3*120**2*4*induct*c1)#
print "ripple voltage   =   %0.3f"%((v1)),"volt"
#(2) with two filter
v1=4*voltag/((3*3.14**5)*(16*120**2*induct**2*c1**2))#
print "ripple voltage including filters   =   %0.2f"%((v1)),"volt"#correction in the book
#(3)ripple voltage
v1=4*voltag/(5*3.14*1.414*2*3.14*240*240*3.14*induct*c1)#
v1=v1/20#
print "ripple voltage   =   %0.4f"%((v1)),"volt"

from __future__ import division
from math import sqrt
voltag=375##volt
r1=2000##ohm
induct=20##henry
c1=16*10**-6##farad
r11=100##ohm
r=200##ohm
#(1) voltage and ripple with load
print "voltage and ripple with load"
r=r+r11+400#
vdc=((2*sqrt(2)*voltag/3.14))/1.35#
ripple=r1/(3*sqrt(2)*(377)*induct*2)#
print "vdc   =   %0.2f"%((vdc)),"volt"
print "ripple   =   %0.2e"%((ripple))
#(2) capacitance connected across load
print "capacitance connected across load"
vdc=sqrt(2)*voltag/(1+1/(4*(60)*r1*2*c1))#
ripple=1/(4*sqrt(3)*(60)*r1*2*c1)#
print "vdc   =   %0.2f"%((vdc)),"volt"
print "ripple   =   %0.2e"%((ripple))
#(3) filter containing two inductors and capacitors in parallel
print "filter containing two inductors and capacitors in parallel"
vdc=250##volt
ripple=0.83*10**-6/(2*induct*2*c1)##correction in the book
print "vdc   =   %0.2f"%((vdc)),"volt"
print "ripple   =   %0.2e"%((ripple))
#(4) two filter
print "two filter"
vdc=250#
ripple=sqrt(2)/(3*16*3.14**2*60**2*induct*c1)**2##correction in the book
print "vdc   =   %0.2f"%((vdc)),"volt"
print "ripple   =   %0.2e"%((ripple))
vdc=sqrt(2)*voltag/(1+(4170/(r1*16))+(r/r1))#
ripple=3300/(16**2*2*20*r1)#
print "vdc   =   %0.2f"%((vdc)),"volt"
print "ripple   =   %0.2e"%((ripple))

from math import sqrt
capaci=4##farad
induct=20##henry
i1=50*10**-3##ampere
resist=200##ohm
maxvol=300*sqrt(2)#
vdc=maxvol-((4170/capaci)*(i1))-(i1*resist)#
ripple=(3300*i1)/((capaci**2)*(induct)*353)#
print "output voltage   =   %0.2f"%((vdc)),"volt"
print "ripple voltage   =   %0.2e"%((ripple))

from math import sqrt
voltag=25##volt
c1=10*10**-6##farad
i1=100*10**-3##ampere
ripple=0.001#
w=754##radians
#(1) inductance and resistance


r1=voltag/i1#
induct=40/(sqrt(2)*w**2*(c1))#
print "inductance of filter   =   %0.2f"%((induct)),"henry"#correction in the book
print "resistance of filter   =   %0.2f"%((r1)),"ohm"

from math import exp
resacu=0.1*10**-12##ampere
u=20+273##kelvin
voltaf=0.55##volt
w=1.38*10**-23#
q=1.6*10**-19#
for z in range(1,3):
    if z==2 :
        u=100+273#
        print "current at 100celsius rise"
    
    voltag=w*u/q#
    i1=(10**-13)*(exp((voltaf/voltag))-1)#
    if z==2:
        i1=(256*10**-13)*((exp(voltaf/voltag)-1))#
    
    print "current   =   %0.2e"%((i1)),"ampere"

from math import log
na=10*22##atoms per cubic metre
nd=1.2*10**21##donor per cubic metre
voltag=1.38*10**-23*(273+298)/(1.6*10**-19)##correction in the book
voltag=0.026#
ni=1.5*10**16#
ni=ni**2#
v1=voltag*log((na*nd)/(ni))#
print "thermal voltage   =   %0.3f"%((voltag)),"volt"
print "barrier voltage   =   %0.3f"%(abs(v1)),"volt"#correction in the book

from math import exp
i1=2*10**-7##ampere
voltag=0.026##volt
i=i1*((exp(0.1/voltag)-1))#
print "current   =   %0.2e"%((i)),"ampere"

from math import exp
resacu=1*10**-6##ampere
voltaf=150*10**-3##volt
w=8.62*10**-5#
voltag=0.026##volt
u=300##kelvin
uw=u*w#
resist=(uw)/((resacu)*exp(voltaf/voltag))#
print "resistance at 150mvolt   =   %0.2f"%((resist)),"ohm"#correction in the book

from math import log
dopfac=1000#
w=300##kelvin
q=0.026*log(dopfac)#
print "change in barrier   =   %0.2f"%((q)),"volt"

from math import sqrt
area12=1*10**-8##metre square
volre1=-1##reverse voltage
capac1=5*10**-12##farad
volbu1=0.9##volt
voltag=0.5##volt
i1=10*10**-3##ampere
durmin=1*10**-6##ssecond
#(1) capacitance
capac1=capac1*sqrt((volre1-volbu1)/(voltag-volbu1))#
print "depletion capacitance   =   %0.2e"%((capac1)),"farad"
#(2) capacitance
capac1=i1*durmin/(0.026)#

print "capacitance   =   %0.2e"%((capac1)),"farad"

from math import log
quantg=4*10**22##atoms per cubic centimetre
quants=5*10**22##atoms per cubic centimetre
w=2.5*10**13##per cubic centimetre
w1=1.5*10**10##per cubic centimetre
for q in [quantg, quants]:
    na=2*q/(10**8)
    nd=500*na#
    if q==quantg :
        w=w#
        voltag=0.026*log(na*nd/w**2)#
        print "potential germanium   =   %0.2f"%((voltag)),"volt"
    
    if q==quants:
        w=w1#
        voltag=0.026*log(na*nd/w**2)#
        print "potential silicon   =   %0.2f"%((voltag)),"volt"

u=0.05##metre square per velocity second correction in the book
un=0.13##metre square per velocity second
condun=20##second per metre conductivity of n region
condup=1000##second per metre conductivity of p region
p=condup/(1.6*10**-19*u)#
no=condun/(1.6*10**-19*un)#
print "electrons density   =   %0.2e"%((no)),"per cubic metre"
print "holes density   =   %0.2e"%((p)),"per cubic metre"#others to find is not in the book

