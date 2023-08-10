colcur=10*10**-3##ampere
vce=10##volt
hie=500##ohm
hoe=4*10**-5
hfe=100
hre=1*10**-4
fqu=50*10**6##hertz
q=3*10**12##farad
voltag=26*10**-3##volt
g=colcur/voltag
gbe=g/hfe
gbc=gbe*hre
rbb=hie-260
oucond=hoe-(1+hfe)*gbc
cbe=g/(2*3.14*fqu)
rbc=1/gbc
rce=1/oucond
print "transconductance g   =   %0.2f"%((g)),"ampere/volt"
print "input conductance gbe   =   %0.2e"%((gbe)),"ampere/volt"
print "feedback conductance gbc   =   %0.2e"%((gbc)),"ampere/volt"
print "base spread resistance rbb   =   %0.2f"%((rbb)),"ohm"
print "output conductance   =   %0.2e"%((oucond)),"ampere/volt"
print "transition capacitance cbe   =   %0.2e"%((cbe)),"farad"
print "rbc    =   %0.2e"%((rbc)),"ohm"##correction as 2.6mega ohm
print "rce   =   %0.2e"%((rce)),"ohm"

colcur=5*10**-3##ampere
vce=10##volt
hfe=100
hie=600##ohm
cugain=10
fqu=10*10**6##hertz

tracat=3*10**-12##farad
voltag=26*10**-3##volt
fbeta1=((((hfe**2)/(cugain**2))-1)/fqu**2)**(1/2)
fbeta1=1/fbeta1
fq1=hfe*fbeta1
cbe=colcur/(2*3.14*fq1*voltag)
rbe=hfe/(colcur/voltag)
rbb=hie-rbe
print "fbeta   =   %0.2f"%((fbeta1)),"hertz"
print "f   =   %0.2f"%((fq1)),"hertz"
print "cbe   =   %0.2e"%((cbe)),"farad"
print "rbe   =   %0.2f"%((rbe)),"ohm"
print "rbb   =   %0.2f"%((rbb)),"ohm"

w=1*10**-4##centimetre
em1cur=2*10**-3##ampere
q=47
voltag=26*10**-3##volt
cde=(em1cur*w**2)/(voltag*2*q)
fq1=(em1cur)/(2*3.14*cde*voltag)
print "cde   =   %0.2e"%((cde)),"farad"
print "frequency   =   %0.2e"%((fq1)),"hertz"

w=5*10**-4##centimetre
em1cur=2*10**-3##ampere
q=47
voltag=26*10**-3##volt
re=voltag/em1cur
fq1=2*q/(w**2*2*3.14)
cde=(em1cur*w**2)/(voltag*2*q)
w=(w**2)/(2*q)
print "re   =   %0.2f"%((re)),"ohm"
print "falpha   =   %0.2e"%((fq1)),"hertz"
print "cde   =   %0.2e"%((cde)),"farad"
print "w   =   %0.2e"%((w)),"second"

w=10**-6##centimetre
em1cur=4*10**-3##ampere
voltag=26*10**-3##volt
q=47
cde=(em1cur*w**2)/(voltag*2*q)
fq1=(em1cur)/(2*3.14*cde*voltag)
print "f   =   %0.2e"%((fq1)),"hertz"
print "cde   =   %0.2e"%((cde)),"farad"##correction required in the book.
