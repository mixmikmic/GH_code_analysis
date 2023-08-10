import math

#Variables

ID = 5.0 * 10**-3              #Drain current (in Ampere)
VDD = 10.0                     #Voltage (in volts)
RD = 1.0 * 10**3               #Drain resistance (in ohm)
RS = 500.0                     #Source resistance (in ohm)                    

#Calculation

VS = ID * RS                   #Source voltage (in volts)
VD = VDD - ID * RD             #Drain voltage (in volts)
VDS = VD - VS                  #Drain-Source voltage (in volts)
VGS = -VS                      #Gate-to-source voltage (in volts)

#Result

print "Value of drain-to-source voltage is ",VDS," V.\nValue of Gate-to-source voltage is ",VGS," V."

import math

#Variables

RD = 56.0 * 10**3                  #Drain resistance (in ohm)
RG = 1.0 * 10**6                   #Gate resistance (in ohm)
IDSS = 1.5 * 10**-3                #Drain to ground current (in Ampere)
Vp = -1.5                          #Voltage (in volts)
VDD = 20.0                         #Supply voltage (in volts)
VD = 10.0                          #Drain voltage (in volts)  
R = 4.0 * 10**3                    #Resistance (in ohm)  

#Calculation

ID = (VDD - VD) / RD               #Drain current (in Ampere) 
VGS = (1 - (ID / IDSS)**0.5)*Vp    #Gate-to-source voltage (in volts)
VS = -VGS                          #Source voltage (in volts)  
R1 = VS / ID - R                   #Resistance R1 (in ohm)

#Result

print "Value of resistance R1 is ",round(R1 * 10**-3,1)," kilo-ohm."

import math

#Variables

ID = 1.5 * 10**-3                  #Drain current (in Ampere)
IDSS = 5.0 * 10**-3                #Drain-to-source current (in Ampere)     
Vp = -2.0                          #Voltage (in volts)
VDS = 10.0                         #Drain-to-source voltage (in volts)
VDD = 20.0                         #Supply voltage (in volts)  

#Calculation

VGS = (1 - ID/IDSS)*Vp             #Gate-to-Source voltage (in volts)
VS = -VGS                          #Source voltage (in volts)
RS = VS / ID                       #Source resistance (in ohm)
RD = (VDD - VDS) / ID - RS         #Drain resistance (in ohm)

#Result

print "Value of RS is ",round(RS)," ohm.\nValue of RD is ",round(RD * 10**-3,1)," kilo-ohm."

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,title,xlabel,ylabel,annotate

#Variables

RD = 1.8 * 10**3                  #Drain resistance (in ohm)
RS = 270.0                        #Source resistance (in ohm)
RG = 10.0 * 10**6                 #Resistance (in ohm)
IDSS = 12.0 * 10**-3              #Drain-to-source current (in Ampere) 
ID = 6.0 * 10**-3                 #Drain current (in Ampere)
VDD = 15.0                        #Supply voltage (in volts)

#Calculation

VS = ID * RS                      #Source voltage (in volts)
VGS = - VS                        #Gate-to-Source Voltage     
IDQ = 5.0 * 10**-3                #Drain current at Q point (in Ampere)
VGSQ = -1.4                       #Gate-to-source voltage (in volts)   
VD = VDD - IDQ * RD               #Drain voltage (in volts)

#Graph

x = numpy.linspace(-4,0,100)
plot(x,12*(1 + x/4)**2,'r')
title("transfer characteristic")
plot(x,x*(-5.0/1.40),'b')
xlabel("Gate-to-Source voltage VGS (V)")
ylabel("Drain current ID(mA)")
annotate("Q",xy=(-1.62,6.0))

#Result

print "Quiescent values of ID and VGS is ",IDQ * 10**3," mA and ",VGSQ," V."
print "D.C. voltage between drain and ground is ",VD," V."

import math

#Variables

VP = VGSoff = 5.0                   #Voltage (in volts)
IDSS = 12.0 * 10**-3                #Drain-to-source current (in Ampere)
VDD = 12.0                          #Drain voltage (in volts)
ID = 4.0 * 10**-3                   #Drain current (in Ampere)
VDS = 6.0                           #Drain-to-source voltage (in volts)

#Calculation

VGS = (1 - (ID / IDSS)**0.5)*VGSoff #Gate-to-source voltage (in volts)
VS = VGS                            #Source voltage (in volts)
RS = VS / ID                        #Source resistance (in ohm)
RD = (VDD - VDS) / ID               #Drain resistance (in ohm)

#Result

print "Value of RD is ",RD * 10**-3," kilo-ohm.\nValue of RS is ",round(RS)," ohm."

#Slight variation due to higher precision.

import math

#Variables

IDSS = 10.0 * 10**-3                    #Drain-to-source current (in Ampere)
VDD = 20.0                              #Drain voltage (in volts)

#Calculation

IDQ = IDSS / 2                          #Drain current at Q point (in Ampere)
VDSQ = VDD / 2                          #Drain-to-source voltage at Q point (in volts)
VGS = -2.2                              #Gate-to-source voltage (in volts)
ID = 5.0 * 10**-3                       #Drain current (in Ampere)
RD = (VDD - VDSQ) / ID                  #Drain resistance (in ohm)
VS = - VGS                              #Source voltage (in volts)
RS = VS / ID                            #Source resistance (in ohm)      

#Result

print "Operating point is ID = ",IDQ * 10**3," mA and VDS = ",VDSQ," V."
print "Value of RD is ",RD * 10**-3," kilo-ohm and RS is ",RS," ohm."

import math

#Variables

VDD = 20.0                        #Supply voltage (in volts)
RD = 2.5 * 10**3                  #Drain resistance (in ohm)
RS = 1.5 * 10**3                  #Source resistance (in ohm)
R1 = 2.0 * 10**6                  #Resistance (in ohm)
R2 = 250.0 * 10**3                #Resitance (in ohm)
ID = 4.0 * 10**-3                 #Drain current (in Ampere) 

#Calculation

VG = VDD * R2 / (R1 + R2)         #Gate voltage (in volts)
VS = ID * RS                      #Source voltage (in volts)
VGS = VG - VS                     #Gate-to-source voltage (in volts)
VD = VDD - ID * RD                #Drain voltage (in volts)   

#Result

print "Value of VGS is ",round(VGS,1)," V. and value of VDS is ",VD - VS," V."

import math

#Variables

gm = 4.0 * 10**-3                      #Transconductance (in Siemen)
RD = 1.5 * 10**3                       #Drain resistance (in ohm) 

#Calculation

Av = -gm * RD                          #Voltage gain    

#Result

print "Voltage gain is ",Av,"."

import math

#Variables

gm = 2.5 * 10**-3                      #Transconductance (in Ampere per volt)
rd = 500.0 * 10**3                     #Resistance (in ohm)
RD = 10.0 * 10**3                      #Load resistance (in ohm)

#Calculation

rL = RD * rd / (RD + rd)               #a.c. equivalent resistance (in ohm)
Av = -gm * rL                          #Voltage gain   

#Result

print "Voltage gain is ",round(Av,1),"."

import math

#Variables

gm = 2.0 * 10**-3                      #Transconductance (in Ampere per volt)
rd = 40.0 * 10**3                      #Resistance (in ohm)
RD = 20.0 * 10**3                      #Drain resistance (in ohm)
RG = 100.0 * 10**6                     #Gate resistance (in ohm)    

#Calculation

rL = RD * rd / (RD + rd)               #a.c. equivalent resistance (in ohm)
Av = -gm * rL                          #Voltage gain   
R1i = RG                               #input resistance (in ohm)
R1o = rL                               #output resistance (in ohm) 

#Result

print "Voltage gain is ",round(Av,1),"."
print "Input resistance is ",R1i * 10**-6," Mega-ohm.\nOutput resistance is ",round(R1o * 10**-3,1)," kilo-ohm." 

import math

#Variables

gm = 2.0 * 10**-3                           #Transconductance (in Ampere per volt)
rd = 10.0 * 10**3                           #Resistance (in ohm)
RD = 50.0 * 10**3                           #Drain resistance (in ohm)

#Calculation

rL = RD * rd / (RD + rd)                    #a.c. equivalent resistance (in ohm)
Av = - gm * rL                              #Voltage gain

#Result

print "Voltage gain is ",round(Av,2),"."

import math

#Variables

RD = 100.0 * 10**3                     #Drain resistance (in ohm)              
gm = 1.6 * 10**-3                      #Transconductance (in Ampere per volt)
rd = 44.0 * 10**3                      #Resistance (in ohm)
Cgs = 3.0 * 10**-12                    #Capacitance gate-to-source (in Farad)
Cds = 1.0 * 10**-12                    #Capacitance drain-to-source (in Farad)
Cgd = 2.8 * 10**-12                    #Capacitance gate-to-drain (in Farad) 

#Calculation

rL = RD * rd / (RD + rd)               #a.c. load resistance (in ohm)  
Av = -gm * rL                          #Voltage gain 

#Result

print "Voltage gain is ",round(Av,1),'.'

import math

#Variables

gm = 4500.0 * 10**-6                   #Transconductance (in Ampere per volt)
RD = 3.0 * 10**3                       #Drain resistance (in ohm)
RL = 5.0 * 10**3                       #Load  resistance (in ohm) 
Vin = 100.0 * 10**-3                   #Input voltage (in volts)
ID = 2.0 * 10**-3                      #Drain current (in Ampere)

#Calculation

rL = RD * RL / (RD + RL)               #a.c. load resistance (in ohm)
vo = -gm * rL * Vin                    #Output voltage (in volts)   

#Result

print "Output voltage is ",abs(round(vo,3))," V."

import math

#Variables

gm = 4.0 * 10**-3                     #Transconductance (in Siemen)
RD = 1.5 * 10**3                      #Drain resistance (in ohm)
RG = 10.0 * 10**6                     #Gate resistance (in ohm)
rs = 500.0                            #resistance (in ohm)     

#Calculation

#Voltage gain when Rl is zero

rL = RD                               #a.c. load resistance (in ohm)
Av = -(gm * rL)/(1 + gm * rs)         #Voltage gain1 

#Voltage gain when Rl is 100 kilo-ohm

RL = 100.0 * 10**3                    #Load resistance (in ohm)      
rL1 = RD * RL / (RD + RL)             #a.c. load resistance (in ohm)
Av1 = -(gm * rL1)/(1 + gm * rs)       #Voltage gain1 

#Result

print "Voltage gain when RL is zero is ",Av,".\nVoltage gain when Rl is 100 kilo-ohm is ",round(Av1,2),"." 

import math

#Variables

RD = 1.5 * 10**3                        #Drain resistance (in ohm)
RS = 750.0                              #Source resistance (in ohm)
RG = 1.0 * 10**6                        #Gate resistance (in ohm)
IDSS = 10.0 * 10**-3                    #Supply current (in Ampere)
Vp = -3.5                               #Voltage (in volts)
IDQ = 2.3 * 10**-3                      #Drain current at Q point (in Ampere)
VGSQ = -1.8                             #Gate-to-source voltage at Q point (in volts)

#Calculation

gmo = -2 * IDSS / Vp                    #Maximum transconductance (in Ampere per volt)
gm = gmo * (1 - VGSQ/Vp)                #Transconductance at Q point (in Ampere per volt)               
rL = RD                                 #a.c. load resistance (in ohm)
Av = - gm * rL / (1 + gm * RS)          #Unbypassed RS (in ohm)
Av1 = -gm * rL                          #Bypassed RS (in ohm) 

#Result

print "Voltage gain for unbypassed Rs is ",round(Av,2),".\nVoltage gain for bypassed Rs is ",round(Av1,3),"."

#Slight variation due to higher precision

import math

#Variables

gm = 8000.0 * 10**-6                   #Transconductance (in Siemen)
RS = 10.0 * 10**3                      #Drain resistance (in ohm)
RG = 100.0 * 10**6                     #Gate resistance (in ohm) 

#Calculation

Av = RS / (RS + 1 / gm)                #Voltage gain
R1i = RG                               #Input resistance (in ohm)
R1o = 1 / gm                           #Output resistance (in ohm)

#Result

print "Voltage gain is ",round(Av,3),".\nInput resistance is ",R1i * 10**-6," Mega-ohm.\nOutput resistance is ",R1o," ohm."

import math

#Variables

vin = 2.0 * 10**-3                      #Input voltage (in volts)
gm = 5500.0 * 10**-6                    #Transconductance (in Siemen)
R1 = R2 = 1.0 * 10**6                   #Resistance (in ohm)
RS = 5.0 * 10**3                        #Source resistance (in ohm)
RL = 2.0 * 10**3                        #Load resistance (in ohm)

#Calculation

Av = RS / (RS + 1/gm)                   #Voltage gain 
R1i = R1 * R2 / (R1 + R2)               #Input resistance (in ohm)
R1o = RS * 1/gm /(RS + 1/gm)            #Output resistance (in ohm)
Vo = RL / (RL + R1o) * Av * vin         #Output voltage (in volts)

#Result

print "Voltage gain is ",round(Av,3),".\nInput resistance is ",R1i * 10**-6," Mega-ohm.\nOutput resistance is ",round(R1o,1)," ohm.\nOutput voltage is ",round(Vo * 10**3,2)," mV."

import math

#Variables

gm = 2500.0 * 10**-6                   #Transconductance (in Amper per volt)
RD = 10.0 * 10**3                      #Drain resistance (in ohm)
RS = 2.0 * 10**3                       #Source resistance (in ohm)

#Calculation

Av = gm * RD                           #Voltage gain 
R1i = RS * 1/gm /(RS + 1/gm)           #Input resistance (in ohm)

#Result

print "Amplifier voltage gain  is ",Av,".\nInput resistance is ",round(R1i)," ohm."

import math

#Variables

gmo = 5.0 * 10**-3                       #Maximum transconductance (in Siemen)
RD = 1.0 * 10**3                         #Drain resistance (in ohm)
RS = 200.0                               #Source resistance (in ohm)
ID = 5.0 * 10**-3                        #Drain current (in Ampere)

#Calculation

R1i = RS * 1/gmo /(RS + 1/gmo)           #Input resistance (in ohm)
VS = ID * RS                             #Source voltage (in volts) 
VGS = VS                                 #Gate-to-Source voltage (in volts)
IDSS = 2 * ID                            #Supply current (in Ampere)
VGSoff = -2 * IDSS / ID                  #Gate-to-source cut off voltage (in volts)
gm = gmo * (1 - abs(VGS / VGSoff))       #Transconductance (in Siemen) 
Av = gm * RD                             #Voltage gain  

#Result

print "Input resistance is ",R1i," ohm.\na.c. voltage gain is ",Av,"."

