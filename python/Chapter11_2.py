import math

#Variable Declaration
R1min=500          #Minimum Value of R1(ohm)
R1max=5*10**3      #Maximum Value of R1(ohm)
C=300*10**-9       #in farad(C=C1=C2) 

#Calculation
#Using the formula f=1/2*pi*R*C for Wein bridge oscillator

fmin=1/(2*math.pi*C*R1max)   #Minimum frequency occurs when R1 is maximum(Hz)
fmax=1/(2*math.pi*C*R1min)   #Maximum frequency occurs when R1 is minimum(Hz)

print "Mimimum frequency f(min)=",round(fmin),"Hz"
print "Maximum frequency f(max)=",round(fmax/1000,2),"kHz"

import math

#Variable Declaration

Vi=5                        #Input voltage(V)
Ib=500*10**-9               #Bias Current(A)

#Calculation
#With R1 and R2 in the circuit
Vr3=0.1                     #As range is 0-0.1V
Vr=Vi-Vr3                   #KVL

I3=100*10**-6               #Since I3>>Ib, assume I3=100micro ampere
R3=Vr3/I3                   #Ohm's Law 
Rr=Vr/I3                    #Ohm's Law. Rr is equivalent series resistance. Rr=R1+R2

print "R3=",round(R3*10**-3),"kilo ohm"
print "R1+R2=",round(Rr*10**-3),"kilo ohm"


#With R2 swithed out of the circuit
Vr3=1                       #Range 0-1V
I3=Vr3/R3                   #Ohm's Law 
Vr1=Vi-Vr3                  #KVL
R1=Vr1/I3                   #Ohm's Law
R2=Rr-R1                    #Rr is equivalent series resistance                       
print "R1=",R1*10**-3,"kilo ohm"
print "R2=",R2*10**-3,"kilo ohm"

import math

#Variable Declaration
C1=0.1*10**-6                  #in farad    
R1=1*10**3                     #in ohm
R2=10*10**3                    #in ohm 
UTP=3.0                        #in V
LTP=-3.0                       #in V
Vcc=15.0                       #in V

#Calculation

V3=Vcc-1                       #Op-amp saturation voltage is approximately one less than Vcc

#For contact at top of R1
V1=V3                          
I2=V1/R2
dV=UTP-LTP
t=C1*dV/I2                     #Using equation for a capacitor charging linearly
f=1/(2*t)

print "For contact at top of R1,"
print "f=",round(f*10**-3,2),"kHz"

#For R1 at 10% from bottom

V1=0.1*V3
I2=V1/R2
t=C1* dV/I2                    #Using equation for a capacitor charging linearly
f=1/(2*t)

print 
print "For R1 contact at 10% from bottom,"
print "f=",round(f),"Hz"

import math

#Variable Declaration
R1=20*10**3             #in ohm
R2=6.2*10**3            #in ohm
R3=5.6*10**3            #in ohm
C1=0.2*10**-6           #in farad
Vcc=12.0                #in volt

#Calculation

Vo=Vcc-1                 #Op-amp saturation voltage is approximately one less than Vcc

UTP=Vo*R3/(R3+R2)        #Upper Threshold Voltage
LTP=-UTP                 #Lower Threshold voltage              
 
t=C1*R1*math.log((Vo-LTP)/(Vo-UTP))    #Equation to find pulse width for astable multivibrator
f=1/(2*t)                               

#Results
print "t=",round(t*10**3,2),"ms"
print "The frequency of the sqaure wave output is ",round(f),"Hz"

import math

#Variable Declaration

Vcc=10
Vb=1
R1=22*10**3
R2=10*10**3
C1=100*10**-12
C2=0.01*10**-6

#Calculation
Vo_plus=Vcc-1
Vo_minus=-(Vcc-1)

PW=C2*R2*math.log((Vo_plus-Vo_minus)/Vb)
print "Pulse width(PW)=",round(PW*10**6),"micro second"

#When Pw=6ms, C2 is found as follows
PW=6*10**-3
C2=PW/(R2*math.log((Vo_plus-Vo_minus)/Vb))

print "For Pw=6ms, C2 should be",round(C2*10**6,1),"micro farad"

