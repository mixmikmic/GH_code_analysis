from math import log
incaco=1.5*10**16##cubic metre
resist=2*10**3##ohm metre
dopcon=10**20##metre
q=26*10**-3##electron volt
#(1)
w=2.25*10**32/dopcon#
#(3)
shifer=q*log(dopcon/incaco)##shift in fermi level
ni=9*10**32#
#(3)
w1=ni/dopcon#
print "minority concentration   =   %0.2e"%((w)),"per metre square"#
print "shift in fermi   =   %0.2f"%((shifer)),"volt"#
print "minority concentration when n doubled   =   %0.2e"%((w1)),"per cubic metre"

numfre=7.87*10**28##per cubic metre
molity=34.8##square centimetre/velocity second
e=30##volt per centimetre
#(1)
molity=molity*10**-4#q=1.6*10**-19#
conduc=numfre*q*molity#
#(2)
e=e*10**2#
veloci=(molity*e)#
curden=conduc*e#
print "conductivity   =   %0.2e"%((conduc)),"second per metre"#
print "drift velocity   =   %0.2f"%((veloci)),"metre per second"#
print "density   =   %0.2e"%((curden)),"ampere per cubic metre"

ni=2.5*10**13##per square centimetre
moe=3800#square centimetre/velocity second
mo1=1800##square centimetre/velocity second
num=4.51*10**22##number of atoms
q=1.6*10**-19#
conduc=ni*q*(moe+mo1)#
num=num/10**7#
impura=(ni**2)/num#
ni=5*10**14#
condu1=ni*q*moe#
print "conductivity   =   %0.4f"%((conduc)),"second per centimetre"#
print "conductivity at extent of 1 impurity   =   %0.2f"%((condu1)),"second per centimetre"##there is mistake in book as 3.04s/cm
conduc=num*q*mo1#
print "conductivity  acceptor to extent of 1 impurity   =   %0.2f"%((conduc)),"second per centimetre"

ni=1.5*10**10##per cubic centimetre
moe=1300##square centimetre/velocity second
mo1=500##square centimetre/velocity second
w=5*10**22##atoms per cubic centimetre
q=1.6*10**-19#
#(a) conductivity intrinisc at 300kelvin
conduc=ni*q*(moe+mo1)##conductivity
u=((ni)/(5*10**14))#
ni=5*10**14#
#(b)conductivity when donor atom added to extent of 1 impurity
condu1=ni*q*moe#
print "conductivity intrinisc at 300kelvin   =   %0.2e"%((conduc)),"second per centimetre"#
print "conductivity when donor atom added to extent of 1 impurity   =   %0.3f"%((condu1)),"second per centimetre"#
#conductivity when acceptor added to extent of 1 impurity
conduc=ni*q*mo1#
print "conductivity when acceptor added to extent of 1 impurity   =   %0.3f"%((conduc)),"second per centimetre"

ni=2.5*10**13##per cubic centimetre
moe=3800##square centimetre/velocity second
mo1=1800##square centimetre/velocity second
w=4.5*10**22##atoms per cubic centimetre
q=1.6*10**-19#
#(1) conductivity intrinisc at 300kelvin
conduc=ni*q*(moe+mo1)#
u=10**6#
u=((w)/(u))#
#(2) conductivity with donor impurity 1
condu1=u*q*moe#
print "conductivity intrinisc at 300kelvin   =   %0.3f"%((conduc)),"second per centimetre"#
print "conductivity with donor impurity 1   =   %0.2f"%((condu1)),"second per centimetre"#
u=10**7#u=w/u#
#(3) conductivity with acceptor impurity 1
conduc=u*q*mo1#
print "conductivity with acceptor impurity 1   =   %0.2e"%((conduc)),"second per centimetre"#
u=0.9*(w/10**6)#
#(4) conductivity on both
conduc=u*q*moe#
print "conductivity on both   =   %0.2f"%((conduc)),"second per centimetre"#

ferlev=0.3##electron volt
u=300##kelvin
u1=330##kelvin
ferlev=ferlev*u1/u#
print "fermi   =   %0.2f"%((ferlev)),"electron volt"#
print "fermi below the conduction band"

from math import log
ferlev=0.02##electron volt
q=4##donor impurity added
w=0.025##electron volt
ferlev=-((log(q)-8))/40#
print "fermi   =   %0.2f"%((ferlev)),"electron volt"

from sympy import symbols, solve
area=1.5*10**-2##centimetre square
w=1.6##centimetre
resist=20##ohm centimetre
durati=60*10**-6##second in book given as mili
quanti=8*10**15##photons per second


#(1) resistance at each photon gives a electron hole pair
up=1800##centimetre square per velocity second
un=3800##centimetre square per velocity second
q=1.6*10**-19##coulomb
ni=2.5*10**13##per cubic centimetre
sigma1=1/resist#
z1=3800#
z=-sigma1/q#
u=ni**2/up#
#n=poly([(z1) z u],'n')#
n=symbols('n')
expr=z1*n**2+z*n+u
n=solve(expr,n)[1]
n=7.847*10**13##n>ni taken so it is admissible
p1=ni**2/n#
volume=w*area#
nchang=quanti*durati/volume#
pchang=nchang#
sigm11=q*((n+nchang)*un+(pchang+p1)*up)#
resis1=1/sigm11#
r1=resis1*w/area#
print "resistance   =   %0.2f"%((r1)),"ohm"

from __future__ import division
from math import sqrt
moe=1350##square centimetre/velocity second
mo1=450##square centimetre/velocity second
ni=1.5*10**10##per cubic centimetre
concn1=ni*((sqrt(mo1/moe)))##concentration
concne=((ni**2)/(concn1))

print "concentration of electron   =   %0.2f"%((concn1)),"per cubic centimetre"#
print "concentration of holes   =   %0.2f"%((concne)),"per cubic centimetre"#

resist=0.12##ohm metre
q=1.6*10**-19#
concn1=((1/resist)/(0.048*q))##concentration of hole
concne=((1.5*10**16)**(2))/concn1##concentration of electron
print "concentration of hole   =   %0.2e"%((concn1)),"per cubic centimetre"#
print "concentration of electron   =   %0.2e"%((concne)),"per cubic centimetre"

resist=1*10**3##ohm
w=20*10**-6##wide metre
w1=400*10**-6##long metre
mo1=500##square centimetre/velocity second
q=1.6*10**-19#
conduc=(resist*w*4*10**-6)/w1#
concentration=((1)/(conduc*mo1*q))#
print "concentration of acceptor atoms   =   %0.2e"%((concentration)),"per cubic metre"##correction in the book

w=0.026#
moe=3800##square centimetre/velocitysecond
mo1=1300##square centimetre/velocitysecond
u=(moe*w)#
u1=(mo1*w)#
print "dn constants   =   %0.2f"%((u)),"square metre per second"##correction in the book
print "dp constants   =   %0.2f"%((u1)),"square metre per second"##correction in the book

from math import log
w=0.026*(3/2)*log(3)/2#
print "distance of fermi level from center   =   %0.3f"%((w)),"   electron volt"

up=1800##centimetre square per velocity second
un=3800##centimetre square per velocity second

#(1) resistivity is 45 ohm
q=1.6*10**-19##coulomb
ni=2.5*10**13#
sigma1=(un+up)*q*ni#
resist=1/sigma1#
print "resistivity   =   %0.2f"%((resist)),"   ohm centimetre"#
print "resistivity equal to 45"#
#(2) impurity added to extent of 1 atom per 10**9
n=4.4*10**22/10**9
p1=ni**2/n#
sigma1=(n*un+p1*up)*q#
resist=1/sigma1
print "resistivity   =   %0.2f"%((resist)),"   ohm centimetre"#
print "resistivity equal to 32.4"

from math import sqrt
from sympy import symbols, solve, exp
nd=4*10**14##atoms per cubic centimetre
na=5*10**14##atoms per cubic centimetre
#(1) concentration
ni=2.5*10**13#
np=ni**2#
#p1=n+10**14
z=1#
z1=10**14#
u=-ni**2#
#n=poly([z z1 u],'q')#
n=symbols('n')
expr = z*n**2+z1*n+u
n = solve(expr,n)[1]
n=1.05*10**4#
print "concentration of the a free electrons   =   %0.2e"%((n))
p1=n+10**14#
print "concentration of the a free holes   =   %0.2e"%((p1))
#(2)
print "sample p"#
a=ni**2/(300**3*exp(-(0.785/0.026)))#
w=400##kelvin
ni=sqrt(a*w**3*exp(-0.786/(8.62*10**-5*w)))#
ni=((n)*(n+10**14))/10**3#
n=ni-0.05*10**15#
print "n   =   %0.2e"%((n)),"electrons per cubic centimetre"
p1=n+10**14#
print "p   =   %0.2e"%((p1)),"holes per cubic centimetre"

print "essentially intrinsic"

from __future__ import division
w=300##kelvin
conduc=300##ohm centimetre inverse
u=1800#
p=conduc/(u*1.6*10**-19)##concentration holes
n=(2.5*10**13)**2/(p)#
print "concentration of n   =   %0.2e"%((n)),"electrons per cubic centimetre"
print "concentration of holes   =   %0.2e"%((p)),"holes per cubic centimetre"

from __future__ import division
from sympy import symbols, solve
nd=10**14##atoms per cubic centimetre
na=5*10**13##atoms per cubic centimetre
un=3800#
up=1800#
q=1.6*10**-19##coulomb
resist=80##ohm metre
e1=5##volt per metre
w=nd-na#
ni=(un+up)*q*resist#
n=symbols('n')
#p1=oly([1 w -ni**2],'q')#
expr = n**2+w*n-ni**2
##p1=taken as 3.65*19**12
p1=solve(expr, p1)
p1=3.65*10**12#
n=p1+w#
j=(n*un+p1*up)*q*e1#
print "current density   =   %0.2f"%((j)),"ampere per square centimetre"

from __future__ import division
na=1*10**16##per cubic centimetre            correction in the book
ni=1.48*10**10##per cubic centimetre
un=0.13*10**4##centimetre square per velocity second
u=0.05*10**4##centimetre square per velocity second
n=ni**2/na#
q=1/(1.6*10**-19*(un*n+(u*na)))#
print "resistivity   =   %0.2f"%((q)),"ohm centimetre"

from __future__ import division
e1=750##volt per metre
b=0.05##metre square per velocity second
un=0.05##metre square per velocity second
up=0.14##metre square per velocity second
#(1) voltage
w=1.25*10**-2##metre
v1=e1*w#
print "voltage across sample   =   %0.2f"%((v1)),"volt"#
#(2) drift velocity
vd=un*e1#
print "drift velocity   =   %0.2f"%((vd)),"metre per second"#
#transverse force per  coulomb
f1=vd*b#
print "transverse force per  coulomb   =   %0.2f"%((f1)),"newton per coulomb"#
#(4) transverse electric field
e1=vd*b#
print "transverse electric field   =   %0.2f"%((e1)),"volt per metre"#
#(5) hall voltage
q=0.9*10**-2#
vh=e1*q
print "hall voltage   =   %0.2f"%((vh)),"volt"

from __future__ import division
un=1300##centimetre square per velocity second
#at 300kelvin
ni=1.5*10**10#
u=500##centimetre square per velocity second
conduc=1.6*10**-19*1.5*10**10*(un+u)#
q=1/conduc#
#impurity of 1 atom included per 10**5 atoms
print "resistivity at 300kelvin   =   %0.2e"%((q)),"ohm centimetre"#
n=5*10**22/10**5#
p=ni**2/n#
q=1/(1.6*10**-19*(un*n+(u*p)))

print "resistivity at impurity of 1 atom included per 10**5 atoms   =   %0.3f"%((q)),"ohm centimetre"

from __future__ import division
from math import sqrt, log, log10
n=4.4*10**22#
nd=n/10**7#
w=300##kelvin
nc=4.82*10**15*w**(3/2)/1/sqrt(8)#
ec_ef1=-0.026*log((nc/(nd)))#
print "ec-ef   =   %0.2f"%((ec_ef1))
#(2) impurities included inratio 1 to 10**3
n=4.4*10**22#
nd=n/(10**3)#
ec_ef1=-0.026*log(nc/nd)#
print "ec-ef   =   %0.2f"%((ec_ef1)),"electron volt   ef above ec"#
q=log10(nd/nc)/log10(10)#
print "impurities included per germanium atoms   =   0.0002"#

from __future__ import division
from math import log
n=5*10**22##atoms per cubic centimetre
#(1) 1 atom per 10**6
m=0.8##metre
na=n/10**6#
w=300##kelvin
nv=4.82*10**15*(m)**(3/2)*w**(3/2)#
ef_ec=0.026*log(nv/na)#
print "ef-ec   =   %0.2f"%((ef_ec)),"electron volt"#
#(2) impurity included 10*10**3 per atom
na=n/(10*10**3)#
ef_ec=0.026*log(nv/na)#
print "ef-ec   =   %0.2f"%((ef_ec)),"electron volt"#
#(3) condition to concide ec=ef
na=4.81*10**15#
w=(nv/na)**(2/3)#
print "temperature   =   %0.2f"%((w)),"kelvin"##correction in the book

from __future__ import division
#figure is not given in the book
nd=10**7##per cubic centimetre
na=10**17##per cubic centimetre
voltag=0.1*3800*10**-4*1500*3*10**-3#
print "hall voltage   =   %0.2f"%((voltag)),"volt"#
print "remains the same but there change in polarity"

from __future__ import division
vh=60*10**-3##volt
w=6*10**-3##metre
bz=0.1##weber per metre square
i1=10*10**-6##ampere
resist=300000*10**-2##ohm metre
#(1)
#mobility
rh=vh*w/(bz*i1)#
u1=rh/resist#
print "mobilty   =   %0.2f"%((u1)),"metre square per velocity second"

