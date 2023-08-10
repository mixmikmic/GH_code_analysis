from __future__ import division
r=30.8*10**-12## electro optice coefficient in m/V
L=3*10**-2## length in m
y=1.3*10**-6## wavelength in m
n=2.1#
d=30*10**-6## distance between the electrodes in m
V=(y*d)/((n)**3*r*L)## voltage required to have a pi radian phase change in volt
print "The voltage required to have a pi radian phase change  = %0.2f volt"%( V)

a_fc=4## fider cable loss in dB/km
aj=0.7## splice loss in db/km
L=5## length in km
a_cr1=4## connector losses
a_cr2=3.5## connector losses
CL=(a_fc+aj)*L+(a_cr1+a_cr2)## total channel loss in dB
print "The total channel loss =%d dB"%( CL)

from math import sqrt
p=0.5*10**-9## pulse broadening in s/km
L=12## length in km
Pt=p*sqrt(L)## with mode coupling, the total rms broadening in s
BT=20*10**6##
DL=2*(2*Pt*BT*sqrt(2))**4## dispersion equalization penalty in dB
Pt1=p*L## without mode coupling, the total rms broadening in s
DL1=2*(2*Pt1*BT*sqrt(2))**4## without mode coupling, equalization penalty in dB
DL2=2*(2*Pt1*150*10**6*sqrt(2))**4## without mode coupling,dispersion equalization penalty with 125 Mb/s
DL3=2*(2*Pt*125*10**6*sqrt(2))**4## with mode coupling,dispersion equalization penalty with 125 Mb/s
print "with mode coupling, the total rms broadening = %0.2f ns"%( Pt*10**9)#
print "\n The dispersion equalization penalty = %0.2f dB"%( DL*10**4)#
print "\n without mode coupling, the total rms broadening = %0.2f dB"%( Pt1*10**9)#
print "\n without mode coupling, equalization penalty = %0.2f dB"%( DL1)#
print "\n without mode coupling,dispersion equalization penalty with 125 Mb/s = %0.2f dB"%( DL2)#
print "\n with mode coupling,dispersion equalization penalty with 125 Mb/s = %0.2f dB"%( DL3)#
print "\n The answer is wrong in the textbook"

Pi=-2.5## mean optical power launched into the fiber in dBm
Po=-45## mean output optical power available at the receiver in dBm
a_fc=0.35## fider cable loss in dB/km
aj=0.1## splice loss in db/km
a_cr=1## connector losses
Ma=6## safety margin in dB
L=(Pi-Po-a_cr-Ma)/(a_fc+aj)## length in km when system operating at 25 Mbps
Po1=-35## mean output optical power available at the receiver in dBm
L1=(Pi-Po1-a_cr-Ma)/(a_fc+aj)## length in km when system operating at 350 Mbps
print "The length when system operating at 25 Mbps = %0.2f km"%( L)#
print "\n The length when system operating at 350 Mbps = %0.2f km"%( L1)

Tx=-80## transmitter output in dBm
Rx=-40## receiver sensitivity in dBm
sm=32## system margin in dB
L=10## in km
fl=2*L## fider loss in dB
cl=1## detector coupling loss in dB
tl=0.4*8## total splicing loss in dB
ae=5## angle effects & future splice in dB
ta=29.2## total attenuation in dB
Ep=2.8## excess power margin in dB
print "The fider loss = %0.2f dB"%( fl)#
print "\n The total splicing loss = %0.2f dB"%( tl)#
print "\n The fangle effects & future splice = %0.2f dB"%( ae)#
print "\n The total attenuation = %0.2f dB"%( ta)#
print "\n The excess power margin = %0.2f dB"%( Ep)#
print "\n hence the system can operate with small excess power margin"

from math import log
Lc=1## connector loss in db
Ls=5## star coupler insertion loss in dB
af=2## fider loss in dB
Ps=-14## transmitted power in dBm
Pr=-49## receiver sensitivity in dBm
sm=6## system margin in dB
N=16#
L=(Ps-Pr-Ls-4*Lc-(10*log(N))/log(10)-sm)/(2*af)##  max transmission length in km when transmission star coupler is used
N1=32#
L1=(Ps-Pr-Ls-4*Lc-(10*log(N1))/log(10)-sm)/(2*af)## max transmission length in km when reflection star coupler is used
print "The max transmission length when transmission star coupler is used = %0.2f km"%( L)#
print "\n The max transmission length when reflection star coupler is used = %0.2f km"%( L1)

from math import sqrt
y=860*10**-9## wavelength in m
L=5000## length in m
X=0.024#
dy=20*10**-9## spectral width in m
dts=6*10**-9## silica optical link rise time in s
dtr=8*10**-9## detector rise in s
c=3*10**8## speed of light in m/s
dtm=-(L*dy*X)/(c*y)## material dispersion delay time in s
id=2.5*10**-12## intermodel dispersion in s/m
dti=id*L## intermodel dispersion delay time
dtsy=sqrt((dts**2)+(dtr**2)+(dtm**2)+(dti**2))## system rise time in s
Br_max=0.7/dtsy## max bit rate for NRZ coding in bit/s
Br_max1=0.35/dtsy## max bit rate for RZ coding in bit/s
print "The system rise time = %0.2f ns"%( dtsy*10**9)#
print "\n The max bit rate for NRZ coding = %0.2f Mbit/s"%( Br_max/10**6)#
print "\n The max bit rate for RZ coding = %0.2f Mbit/s"%( Br_max1/10**6)

Br=50*10**6## data rate in b/s
c=3*10**8## speed of light in m/s
n1=1.47## 
dl=0.02## 
n12=n1*dl## the difference b/w n1 and n2
L_si=(0.35*c)/(n12*Br)## transmission distance for Si fiber
L_GI=(2.8*c*n1**2)/(2*n1*n12*Br)## transmission distance for GRIN fiber
print "The transmission distance for Si fiber = %0.2f m"%( L_si)#
print "\n The transmission distance for GRIN fiber = %0.2f m"%( L_GI)

Br=20.0*10**6## data rate in b/s
c=3*10**8## speed of light in m/s
y=86*10**-9## wavelength in m
dy=30*10**-9## spectral width in m
X=0.024#
Tb=1/Br#
Lmax=(0.35*Tb*c*y)/(dy*X)## material dispersion limited transmission distance for RZ coding in m
print "The material dispersion limited transmission distance =%d m"%( Lmax)

from __future__ import division
y=860*10**-9## wavelength in m
c=3*10**8## speed of light in m/s
n1=1.47## 
dl=0.02## 
n12=n1*dl## the difference b/w n1 and n2
La=1/1000## loss a in dB/m
Pr=-65## receiver power in dB
Pt=-5## transmitted power in dB
dy=30*10**-9## line width in m
X=0.024#
Lmax=(0.35*c*y)/(dy*X)## material dispersion limited distance for RZ coding in m
L_GI=(1.4*c*n1)/(n12)## model dispersion limited distance for RZ coding in m
L_At=(Pt-Pr)/(La)## attenuation limited distance for RZ coding in m 
print "The material dispersion limited distance = %0.2f*10**10*1/Br m"%( Lmax/10**10)#
print "\n The model dispersion limited distance = %0.2f*10**10*1/Br m"%( L_GI/10**10)#
print "\n The attenuation limited distance =%d-20log(Br) km"%( L_At/10**3)

