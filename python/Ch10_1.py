from __future__ import division
av=1000#
chvoga=0.001##change in voltage gain
beta1=1/((chvoga)/(100/av))-1#
beta1=beta1/av#
fegain=(av)/(1+(av*(beta1)))#
print "reverse transmission   =   %0.2f"%((beta1))
print "gain with feedback   =   %0.2f"%((fegain))

voltag=36##volt
w=0.07##harmonic distortion
inpvol=0.028##volt
beta1=0.012#
a=voltag/inpvol#
fegain=a/(1+beta1*a)##correction in book
volta1=fegain*inpvol#
print "output voltage   =   %0.2f"%((volta1))
#decrease of gain 9
inpvol=9*inpvol#
print "input voltage   =   %0.2f"%((inpvol)),"volt"#

from __future__ import division
volgain=2000##voltage gain
outpower=20##watts
inpsig=10*10**-3##volts
fedbac=40##decibel
fedgai=volgain/100#
outvol=volgain*inpsig##output voltage
inpvol=outvol/fedgai##required input
#10 second harmonic distortion
distor=(10/100)#
print "required input   =   %0.2f"%((inpvol)),"volt"#
print "harmonic distortion   =   %0.2f"%((distor))

from __future__ import division
fedgai=60##decibel
outimp=10*10**3##ohm
outim1=500##ohm modified impedance
fedgai=1000#
fedbac=((outimp/outim1)-(1))/fedgai#
#10 change in gain
overga=1/((1+(fedgai*fedbac))/0.1)##over gain
print "feedback factor   =   %0.3f"%((fedbac))
print "over gain   =   %0.3f"%((overga))

colres=4*10**3##ohm
r=4*10**3##ohm
basres=20*10**3##ohm
r1=1*10**3##ohm
hie=1.1*10**3#
hfe=50#
hoe=(40*10**3)#
ri=basres*hie/(basres+hie)#
curgai=((r1/(r1+ri)))*((basres/(basres+hie)))*((-hfe*colres)/(colres+r))#
volgai=curgai*r/r1#
tranco=volgai/r#
tranre=r1*volgai#
outres=hoe*colres/(hoe+colres)#
print "current gain   =   %0.2f"%((curgai))
print "voltage gain   =   %0.2f"%((volgai))
print "transconductance   =   %0.2f"%((tranco)),"ampere per volt"#
print "transresistance   =   %0.2f"%((tranre)),"ohm"#
print "input resistance   =   %0.2f"%((ri)),"ohm"#
print "output resistance   =   %0.2f"%((outres)),"ohm"#

