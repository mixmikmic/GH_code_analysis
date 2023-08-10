from math import sqrt
freque=5*10**3##hertz
#(1)
g=2*10**-3##ampere per volt
rd=10*10**3##ohm
r1=30*10**3##ohm
r12=r1*r1/(r1+r1)
volgai=-(g*r12*rd)/(r12+rd)
print "voltage gain   =   %0.2f"%((volgai))
#correction : r12 should be taken as 15*10**3ohm in book
#(2) capacitance included
c=0.025*10**-6##farad
frequ1=1/((2*3.14*(((rd*r1)/(rd+r1))+r1))*c)
volgai=(volgai/(sqrt((1+(frequ1/freque)**2))))

print "voltage gain   =   %0.2f"%((volgai))

rd=80*10**3##ohm
r1=8*10**3##ohm
rd12=5*10**3##ohm
rd1=rd*r1/(rd+r1)
u=30
volgai=-(u*rd1)/(rd1+rd12)

print "voltage gain   %0.2f"%((volgai))

r1=60*10**3##ohm
volgai=-17.7
rg=80*10**3##ohm
volgai=((volgai*rg)/(1-volgai))/((rg/(1-volgai))+r1)
print "voltage gain   =   %0.2f"%((volgai))

vds=14##volt
idq=3*10**-3##ampere
vdd=20##volt
g=2*10**-2
rd=50*10**3##ohm
vgs=-1.5##volt
w=(vdd-vds)/idq
r1=-vgs/idq
r2=w-r1
inpres=1/(1-(0.8*((r1)/(r1+r2))))
volgai=(r1+r2)/(r1+r2+(1/(g)))
print "r1   =   %0.2f"%((r1)),"ohm"
print "effective input resistance   =   %0.2f"%((inpres)),"r3ohm"
print "r2   =   %0.2f"%((r2)),"ohm"


print "voltage gain   =   %0.2f"%((volgai)),"av`"

rg=40*10**3##ohm
voltag=(1-6*50)*3.3*10**3/(5.3*10**3)

print "output voltage   =   %0.2f"%((voltag)),"volt"#
#correction required in the book

u=50
rd=10*10**3##ohm
cgs=5*10**-12##farad
cgd=2*10**-12##farad
cds=2*10**-12##farad
freque=3##decibel
g=u/rd
volgai=-u*rd/(rd+rd)
req=rd*rd/(rd+rd)
frequ1=1/(2*3.14*cgd*req)
print "voltage gain   =   %0.2f"%((volgai))
#correction required in book
print "frequency   =   %0.2e"%((frequ1)),"hertz"
capac1=cgd*(1+g)
print "output capacitance   =   %0.2e"%((capac1)),"farad"
print "req   =   %0.2f"%((req)),"ohm"

