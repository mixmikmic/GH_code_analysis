import math
r=1  #in m
p=10**3 
m=4*math.pi*10**-7 #weber/amp-m
c=3*10**8 #speed of light
x=2*math.pi
E_m=(1/r)*(math.sqrt((p*m*c)/x))
print("The value of E in volts/meter= %.5f"%E_m)
B=E_m/c
print("B in weber/meter^2= %.8f"%B)

