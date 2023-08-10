#importing modules
import math
from __future__ import division

#Variable declaration
H=10**6                #Magnetic Field Strength in ampere/m
x=0.5*10**-5           #Magnetic susceptibility 
mu_0=4*math.pi*10**-7

#Calculatiions
M=x*H
B=mu_0*(M+H)

#Result
print"Intensity of Magnetization=",M,"ampere/m"
print"Flux density in the material=",round(B,3),"weber/m^2"


