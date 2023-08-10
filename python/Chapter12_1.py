#importing module
import math
from __future__ import division

#Variable declaration
l=100 #length of resistor in micro-m
w=10 #width of resistor in micro-m
R=0.9 #sheet resistance in k-ohm/n 
End_points=0.65*2 #Total contribution of two end points

#Calculations
Total_squares=l/w
T=Total_squares+End_points #Total effective sqaures
Reff=T*R

#Result
print("Effective Resistance= %0.2f k-ohm" %Reff)

#importing module
import math
from __future__ import division

#Variable declaration
epsilon_0=8.85*10**-14 #in F/cm
epsilon_i=3.9 #in F/cm
tox=0.5*10**-4 #in cm

#Calculations
C=(epsilon_0*epsilon_i)/tox

#Result
print("Capacitance per unit area = %0.2f pF/cm**2" %round(C/10**-8,2))
#The answer provided in the textbook is wrong

#importing module
import math
from __future__ import division

#Variable declaration
Length=4 #in micro-m
Width=1 #in micro-m
R=1000 #in ohm
xj=1*10**-4 #junction depth in cm                

#Calculations
N=Length/Width
R0=R/N
rho=R0*xj

#Result
print("Sheet resistance= %i ohm\n" %R0)
print("average resistivity= %0.3f ohm-cm" %rho)

