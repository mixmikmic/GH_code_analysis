from math import sqrt
d=50*(10**-6) #Core Diameter
n2=1.48 #Core refractive index
n1=1.46 #Cladding refractive index
a=d/2
PI=3.14
#From Example 8.5

n=(n2-n1)/n2

delta=(2*PI*a)/(sqrt(2*n)) #Where delta is the microbending sensor periodicity
delta=round(delta,5)
print "The Microbending Sensor Periodicity is "+str(delta)+" m"

a=5*(10**-7) #Thermal expansion Coefficient
b=6.8*(10**-6) #Thermal Expansion Coefficient
l=1.55*(10**-6) #Given Wavelength
p11=0.126
p12=0.274  #Given Constant Coeffiecients
u=0.17  
n=1.46
#By equation 10.3
dl=l*(a+b) #where dl is the wavelength sensitivity to temp. changes

print "The Wavelength Sensitivity to temparature changes of the filter structure is "+str(dl)+" nm/K"

pe=((n**2)/2)*(((1-u)*p12)-(u*p11))  #where pe is the effective photoelastic coefficient
pe=round(pe,3)

print "The Effective Photoelastic Coefficient is "+str(pe)

#By equation 10.4
dl=l*(1-pe) 

print "As far as Strain is concerned the Sensitivity is %.2e m/e"%(dl)

v=1.5*(10**13) #Raman shift of silcia in terms of Hertz
T=(273+50) #Temperature in terms of Kelvin
d=1 #Fractional change in r in terms of per degree
k=1.38*(10**-23) #Boltzman Constant
h=6.6*(10**-34) #Plancks Constant

#From equation 10.5
dr=(h*v)/(k*(T**2))  #Where dr is the fractional change of temperature sensitivity of raman sensor
dr=dr*100
dr=round(dr,1)
print "The Fractional Change of Temperature Sensitivity of Raman Scattering Sensor is "+str(dr)+" percent per degree celsius"

n=1000 #Turns on the Fibre
r=0.1 #Radius in meter
r2=15 #Earths rotation rate per hour
PI=3.14
c=3*(10**8) #Speed of light
l=1*(10**-6) #Wavelength in meter
r1=(r2*PI)/(180*3600) #Converting it into radian per second

#From equation 10.12
theta=(8*PI*n*PI*(r**2)*r1)/(l*c) #Where theta is the phase shift
theta=round(theta,5)

print "The Phase Shift in Sagnac Gryoscope is "+str(theta)+" radian"

#For Silica
V=4 #in Radian / m T
n=10 #No of turns
I=30 #Current in ampere
ur=1 #Relative permaebility
uo=4*3.14*(10**-7) #Absolute permaebility
PI=3.14
#From equation 10.14
t=PI/180
theta=n*V*uo*ur*I #where theta is the polarization rotation
theta=round(theta,4)

print "The Amount of Polarization rotation is "+str(theta)+" degree"

#Answer miscalculated in Book

