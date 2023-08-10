#zener diode
voltag=5.2##volts
w=260*10**-3##watts
appv=15##voltsw1=50##watts
imax=w/voltag*0.1#
#to maitain a constant voltage
imax1=(w/voltag)-imax#
resmin=(appv-voltag)/(w/voltag)#
resmax=(appv-voltag)/imax1#
#load 50
resmax1=((9.8)/(45*10**-3))-50#
resmin1=((9.8)/(50*10**-3))-50#
res50=resmax1-resmin1#
print "resistance range from %0.2f"%(resmin)," to %0.2f"%(resmax),"ohms"
print "resistance range at 50 from %0.2f"%(resmin1)," to %0.2f"%(resmax1),"ohms"

i1=20*10**-3##ampere
i=30*10**-3##ampere
v1=5.6##volts
v=5.65##volts
#condition
u=35*10**-3##ampere
voltag=5*u+5.5#
print "voltage drop   =   %0.2f"%(voltag)+"volts"

from math import log
v=4.3##volt
q=4##volt
dop=10**17##per cubic centimetre
fi0=0.254*log(dop/(5.1*10**10))#
fi01=0.407+q+0.55#
print 'fi0   =   %0.2f'%(fi01)

from __future__ import division
v1=20##volt
i1=((v1)/(200+1))*10**-3#
print 'current   =   %0.2e'%(i1),'ampere'
#greater than 20
vone=16#
r=vone/i1#
r1=r-1*10**3#
r11=200*10**3-r1#
print 'resistance   =   %0.2e'%(r),'ohm'
print "r1   =   %0.2e"%((r1)),"ohm"
print "r2   =   %0.2e"%((r11)),"ohm"

v1=150##volt
vone=300#volt
idmax=40*10**-3##ampere
idmin=5*10**-3##ampere
r=(vone-v1)/idmax#
imax=idmax-idmin#
print 'maximum current   =   %0.3f'%(imax),'ampere'
#minimum
zq=1#
while (zq<=2):
    if zq==1 :
        ione=25*10**-3#
        i1=ione+idmin#
        vmin=(i1*r)+v1#
        print 'v1 minimum   =   %0.2f'%(vmin),'volt'
    else:
        ione=25*10**-3#
        i1=ione+idmax#
        vmin=(i1*r)+v1#
        print 'v1 maximum   =   %0.2f'%(vmin),'volt'
        
    
    zq=zq+1#
    

from math import sqrt
q=4.5*10**22##atoms per cubic metre
na=q/(10**4)#
eo=0.026*24.16#
e=1.6*10**-19#
W=sqrt((4*16*0.628)/(36*3.14*10**9*na*10**6*e))#
print 'width   =   %0.2e'%(W),'metre'

