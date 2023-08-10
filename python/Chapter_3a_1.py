#importing modules
import math
from __future__ import division

#Variable declaration
N=6.023*10**26
deltaHv=120
B=1.38*10**-23
k=6.023*10**23

#Calculations
n0=0                                          # 0 in denominator
n300=N*math.exp(-deltaHv*10**3/(k*B*300))     #The number of vacancies per kilomole of copper
n900=N*math.exp(-(deltaHv*10**3)/(k*B*900))

#Results
print"at 0K, The number of vacancies per kilomole of copper is",n0
print"at 300K, The number of vacancies per kilomole of copper is",round(n300/10**5,3),"*10**5"
print"at 900K, The numb ber of vacancies per kilomole of copper is",round(n900/10**19,3),"*10**19"

#importing modules
import math
from __future__ import division
from sympy import Symbol

#Variable declaration
F_500=1*10**-10
delta_Hv=Symbol('delta_Hv')
k=Symbol('k')
T1=500+273
T2=1000+273


#Calculations
lnx=math.log(F_500)*T1/T2;
x=math.exp(round(lnx,2))

print"Fraction of vacancies at 1000 degrees C =",round(x*10**7,1),"*10**-7" 

#importing modules
import math
from __future__ import division

#Variable declaration
a=(2*2.82*10**-10)
delta_Hs=1.971*1.6*10**-19
k=1.38*10**-23
T=300

#Calculations
V=a**3                           #Volume of unit cell of NaCl
N=4/V                            #Total number of ion pairs
n=N*math.e**-(delta_Hs/(2*k*T))  

#Result
print"Volume of unit cell of NaCl =",round(V*10**28,3),"*10**-28 m**3"
print"Total number of ion pairs 'N' ='",round(N/10**28,2),"*10**28"
print"The concentration of Schottky defects per m**3 at 300K =",round(n/10**11,2),"*10**11"

#importing modules
import math
from __future__ import division

#Variable declaration
N=6.023*10**23
delta_Hv=1.6*10**-19
k=1.38*10**-23
T=500
mv=5.55;     #molar volume
x=2*10**-8;     #numbber of cm in 1 angstrom

#Calculations
n=N*math.exp(-delta_Hv/(k*T))/mv
a=round(n/(5*10**7*10**6),4)*x;

#Result
print"The number that must be created on heating from 0 to 500K is n=",round(n/10**12,2),"*10**12 per cm**3" #into cm**3
print"As one step is 2 Angstorms, 5*10**7 vacancies are required for 1cm"
print"The amount of climb down by the dislocation is",a*10**8,"cm"

