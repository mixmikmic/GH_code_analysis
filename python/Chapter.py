import math

n2=1.5 #Given Refractive Index of Glass in Air
n1=1   #Given Refractive Index of Air

theta=0 #Brewster's Angle
#From Equation 1.13 (Brewsters angle= Tan Inverse (n2/n1))
theta=math.degrees(math.atan(1.5))
print "The Brewsters Angle of the Material is "+str(round(theta,2))+" Degrees"

n=4 #Total Number of Sources

#For Coherant Sources
print "In Coherant Sources The Maximum Irradiance is "+str(n*n)+"I"  #Where I is the Irradiance at any point
#For Incoherant Sources
print "In Incoherant Sources The Maximum Irradiance is "+str(n)+"I"   

D=0.1 #Diameter of the Objective Lens
d1=500 #Distance from the source
l =0.000000500 #Wavelength Provided
p=1 #First Order
N=40*600 #The diffraction grating is 40 mm wide and has 600 lines/mm

#From Equation 1.29 we Have
Smin=(d1*l)/D  #Where Smin is the minimum Seperation of the Sources
print "(A)The Minimum Seperation Between the Sources is "+str(Smin)+" m"

#We know that Chromatic resolving power is given by l/dl where dl is the Minimum Wavelength Difference
#From Equation l/dl=p*N
dl=l/(N*p)

print "(B)The Minimum Wavelength Difference which may be resolved is "+str(dl)+" m"

em=0.7 #Emissivity Of the Surface
T=2000 #Temperature in Kelvin
A=0.00001 #Area in Meter Square
S=5.67*(10**-8) #Stefan-Boltzmann Constant

W=S*A*em*(T**4) #Where W is the total power radiated

print "The Total Power Radiated from the Source is "+str(W)+" W"

Z=1 #Atomic Number of Hydrogen
m=9.1*(10**-31) #Mass of a Electron
e=1.6*(10**-19) #Charge Of a Electron
p=6.6*(10**-34) #Plancks Constant
e1=8.85*(10**-12)#Permittivity of Free Space
#From Equation 1.43
E=(m*(Z**2)*(e**4))/(8*(p**2)*(e1**2)) #Where E is the Ionization Energy
E2=E/e #Converting in Electron Volts
E2=round(E2,2)

print "The Ionization Energy required to excite the electron from ground to Infinity "+str(E2)+" eV"
       

e=1.6*(10**-19) #Charge Of a Electron
h=6.6*(10**-34) #Plancks Constant
vo=1.1*(10**15) #Threshold Frequency in Hertz

# We Know h*vo=phi*e where phi is the required Work Function
# We assume that the ejected electron has zero kinetic energy

phi=h*vo/e
print "The Required Work function is "+str(phi)+" eV"

