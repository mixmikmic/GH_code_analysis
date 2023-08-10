#importing modules
import math
from __future__ import division
from sympy import Symbol
#Variable declaration
R=Symbol('R')
a=2*R

#Results
print"i)Number of atoms per unit area of (100)plane=",1/a**2
print"ii)Number of atoms per unit area of (110)plane=",1/math.sqrt(2)*a**2
print"iii)Number of atoms per unit area of (111)plane=",1/math.sqrt(3)*a**2

#importing modules
import math
from __future__ import division

#Variable declaration
a=3.61*10**-7
BC=math.sqrt(2)/2
AD=(math.sqrt(6))/2
#Result
print"i)Surface area of the face ABCD =",round(a**2*10**14),"*10**-14 mm**2"
print"ii)Surface area of plane (110) =",round((2/(a*math.sqrt(2)*a)/10**13),2),"*10**13 atoms/mm**2"
print"iii)Surface area of pane(111)=",round(2/(BC*AD*a**2)*10**-13,3),"*10**13 atoms/mm**2"

#importing modules
import math
from __future__ import division

#Variable declaration
h1=1
k1=0
l1=0
h2=1
k2=1
l2=0
h3=1
k3=1
l3=1
a=1

#Calculations
d1=a/(math.sqrt(h1**2+k1**2+l1**2))
d2=a/(math.sqrt(h2**2+k2**2+l2**2))
d3=a/(math.sqrt(h3**2+k3**2+l3**2))

#Result
print"d1 =",d1 
print"d2 =",round(d2,3)
print"d3 =",round(d3,3)
print"d1:d2:d3 =",d1,":",round(d2,3),":",round(d3,3)

#importing modules
import math
from __future__ import division

#Variable declaration
h=2
k=2
l=0
a=450

#Calculations
d=a/(math.sqrt(h**2+k**2+l**2))

#Result
print"d(220) =",round(d,1),"pm"

#importing modules
import math
from __future__ import division

#Variable declaration
a=3.615
r=1.278
h=1
k=1
l=1

#Calculations
a=(4*r)/math.sqrt(2)
d=a/(math.sqrt(h**2+k**2+l**2))

#Result
print"a =",round(a,3),"Angstroms"
print"d =",round(d,3),"Angstroms"

#importing modules
import math
from __future__ import division

#Variable declaration
n=1
lamda=1.54
theta=32*math.pi/180
h=2
k=2
l=0

#Calculations
d=(n*lamda*10**-10)/(2*math.sin(theta))   #derived from 2dsin(theta)=n*l
a=d*(math.sqrt(h**2+k**2+l**2))

#Results
print"d =",round(d*10**10,2),"*10**-10 m"
print"a =",round(a*10**10,1),"*10**-10 m"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=0.58
theta1=6.45*math.pi/180
theta2=9.15*math.pi/180
theta3=13*math.pi/180

#Calculations
dbyn1=lamda/(2*(math.sin(theta1)))
dbyn2=lamda/(2*math.sin(theta2))
dbyn3=lamda/(2*math.sin(theta3))
           
#Results
print"i.  d/n =",round(dbyn1,3),"Angstroms"
print"ii. d/n =",round(dbyn2,3),"Angstroms"
print"iii.d/n =",round(dbyn3,3),"Angstroms"

#importing modules
import math
from __future__ import division

#Variable declaration
d=1.18
theta=90*math.pi/180
lamda=1.540

#Calculations
n=(2*d*math.sin(theta))/lamda

#Result
print"n =",round(n,2)

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=0.58
theta=9.5*math.pi/180
n=1
d=0.5           #d200=a/math.sqrt(2**2+0**2+0**2)=0.5a
#Calculations
a=n*lamda/(2*d*math.sin(theta))     #2*d*sin(theta)=n*lamda 

#Result
print"a =",round(a,2),"Angstorms"

#importing modules
import math
from __future__ import division

#Variable declaration
lamda=0.842
n1=1
q=(8+(35/60))*(math.pi/180)
n2=3
d=1
#Calculations
#n*lamda=2*d*sin(theta)
#n1*0.842=2*d*sin(q)
#n3*0.842=2*d*sin(theta3)
#Dividing both the eauations, we get
#(n2*lamda)/(n1*lamda)=2*d*math.sin(theta3)/2*d*math.sin(q)
theta3=math.asin((((n2*lamda)/(n1*lamda))*(2*d*math.sin(q)))/(2*d))
d=theta3*180/math.pi;
a_d=int(d);
a_m=(d-int(d))*60

#Result
print"sin(theta3) =",a_d,a_m

#importing modules
import math
from __future__ import division

#Variable declaration
a=3.16
lamda=1.54
n=1
theta=20.3*math.pi/180

#Calculations
d=(n*lamda)/(2*math.sin(theta))
x=a/d                             #let math.sqrt(h**2+k**2+l**2)=x

#Result
print"d =",round(d,2),"Angstorms"
print"sqrt(h**2+k**2+l**2) =",round(x,3)
print"Therefore, h**2+k**2+l**2 =sqrt(2)"
print"h =1, k=1"

## importing modules
import math
from __future__ import division

#Variable declaration
n=4
A=107.87
rho=10500
N=6.02*10**26
h=1;
k=1;
l=1;
H=6.625*10**-34
e=1.6*10**-19
theta=(19+(12/60))*math.pi/180
C=3*10**8
#Calculations
a=((n*A)/(rho*N))**(1/3)*10**10
d=a/math.sqrt(h**2+k**2+l**2)
lamda=2*d*math.sin(theta)
E=(H*C)/(lamda*10**-10*e)

#Result
print"a =",round(a,2),"Angstroms"
print"d =",round(d,2),"Angstroms"
print"lamda =",round(lamda,3),"Angstroms"
print"E =",round(E/10**3),"*10**3 eV"

#importing modules
import math
from __future__ import division

#Variable declaration
a=4.57
h=1
k=1
l=1
lamda=1.52
twotheta=33.5*math.pi/180
r=5                  #radius
#Calculations
d=a/(h**2+k**2+l**2)**(1/2)
sintheta=lamda/(2*d)
X=r/math.tan(twotheta)

#Result
print"d =",round(d,2),"Angstorms"
print"sin(theta)=",round(sintheta,3)
print"X =",round(X,3),"cm"

