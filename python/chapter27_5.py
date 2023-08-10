import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,title,xlabel,ylabel,annotate

#Variables 

VCC = 10.0                        #Source voltage (in volts)
R1 = 10.0                         #Resistance (in kilo-ohm)
R2 = 5.0                          #Resistance (in kilo-ohm)
RC = 1.0                          #Collector resistance (in kilo-ohm)    
RE = 500.0 * 10**-3               #Emitter resistance (in kilo-ohm)  
RL = 1.5                          #Load resistance (in kilo-ohm)
beta = 100.0                      #Common emitter current gain
VBE = 0.7                         #Emitter-to-Base voltage (in volts)

#Calculation

VR2 = VCC * (R2 /(R1 + R2))       #Voltage drop across R2 (in volts)
IEQ = (VR2 - VBE) / RE            #Emitter current (in milli-Ampere)
ICQ = IEQ                         #Collector current (in milli-Ampere)
VCEQ = VCC - ICQ * (RC + RE)      #Collector-to-Emitter voltage (in volts)
rL = RC * RL /(RC + RL)           #a.c. load resistance (in kilo-ohm)
ICsat = ICQ + VCEQ / rL           #Collector current at saturation point (in milli-Ampere) 
VCEsat = 0                        #Voltage at saturation point (in volts)
ICcutoff = 0                      #Collector current at cut off point (in milli-Ampere)
VCEcutoff = VCEQ + ICQ * rL       #Collector-to-emitter voltage at cut-off point (in volts)

#Result

print "Collector current at saturation point is ",round(ICsat,2)," mA.\nCollector-to-emitter voltage at saturation point is ",VCEsat," V."
print "Collector current at cut off point is ",ICcutoff," mA.\nCollector-to-emitter voltage at cut-off point is ",VCEcutoff," V."

#Slight variation due to higher precision.

#Graph

x = numpy.linspace(0,5.27,100)
plot(x,8.78 - 8.78/5.27 * x)
title("a.c. load line")
xlabel("Collector-to-emitter voltage VCE (V)")
ylabel("Collector current IC (mA)")
annotate("Q",xy=(2.11,5.26))
annotate("8.78",xy=(0,8.78))
annotate("5.27",xy=(5.27,0))

import math

#Variables 

VCC = 20.0                        #Source voltage (in volts)
R1 = 10.0                         #Resistance (in kilo-ohm)
R2 = 1.8                          #Resistance (in kilo-ohm)
RC = 620.0 * 10**-3               #Collector resistance (in kilo-ohm)    
RE = 200.0 * 10**-3               #Emitter resistance (in kilo-ohm)  
RL = 1.2                          #Load resistance (in kilo-ohm)
beta = 180.0                      #Common emitter current gain
VBE = 0.7                         #Emitter-to-Base voltage (in volts)

#Calculation

VB = VCC * (R2 /(R1 + R2))        #Voltage drop across R2 (in volts)
VE = VB - VBE                     #Voltage at the emitter (in volts)
IE = VE / RE                      #Emitter current (in milli-Ampere)
IC = IE                           #Collector current (in milli-Ampere)
VCE = VCC - IE*(RC + RE)          #Collector-to-emitter voltage (in volts)
ICEQ = IC                         #Collector current at Q (in milli-Ampere)
VCEQ = VCE                        #Collector-to-emitter voltage at Q (in volts)  
rL = RC * RL/(RC + RL)            #a.c. load resistance (in kilo-ohm)  
PP = 2 * ICEQ * rL                #Compliance of the amplifier (in volts) 

#Result

print "Overall compliance (PP) of the amplifier is ",round(PP,2)," V."

import math

#Variables 

r1e = 8.0                       #a.c. load resistance (in ohm)
RC = 220.0                      #Collector resistance (in ohm)
RE = 47.0                       #Emitter resistance (in ohm)
R1 = 4.7 * 10**3                #Resistance (in ohm)
R2 = 470.0                      #Resistance (in ohm)
beta = 50.0                     #Common emitter current gain

#Calculation

rL = RC                         #Load resistance (in ohm)
Av = rL / r1e                   #Voltage gain
Ai = beta                       #Current gain
Ap = Av * Ai                    #Power gain  

#Result

print "Voltage gain is ",Av," and Power gain is ",Ap,"."

import math

#Variables 

Ptrdc = 20.0                        #dc Power (in watt)
Poac = 5.0                          #ac Power (in watt)     

#Calculation

ne = Poac / Ptrdc                   #Collector efficiency   
P = Ptrdc                           #Power rating of the transistor

#Result

print "Collector efficiency is ",ne * 100,"% .\nPower rating of the transistor is ",P," W."

import math

#Variables 

Pcdc = 10.0                   #dc power (in watt)
ne = 0.32                     #efficiency

#Calculation

Poac = ne * Pcdc / (1 - ne)   #a.c. power output (in watt)

#Result

print "The a.c. power output is ",round(Poac,1)," W."

import math

#Variables 

nc = 0.5                       #Efficiency
VCC = 24.0                     #Source voltage (in volts)
Poac = 3.5                     #a.c. power output (in watt)

#Calculation

Ptrdc = Poac / nc              #dc power (in watt)
Pcdc = Ptrdc - Poac            #Power dissipated as heat (in watt)   

#Result

print "Total power within the circuit is ",Ptrdc," W.\nThe power Pcdc = ",Pcdc," W is dissipated in the form of heat within the transistor collector region."

import math

#Variables 

VCC = 20.0                       #Supply voltage (in volts)
VCEQ = 10.0                      #Collector-to-emitter voltage (in volts)
ICQ = 600.0 * 10**-3             #Collector current (in Ampere)
RL = 16.0                        #Load resistance (in ohm)
Ip = 300.0 * 10**-3              #Output current variation (in Ampere)

#Calculation

Pindc = VCC * ICQ                #dc power supplied (in watt)
PRLdc = ICQ**2 * RL              #dc power consumed by load resistor (in watt)  
I = Ip / 2**0.5                  #r.m.s. value of Collector current (in Ampere) 
Poac = I**2 * RL                 #a.c. power across load resistor (in ohm) 
Ptrdc = Pindc - PRLdc            #dc power delievered to transistor (in watt)
Pcdc = Ptrdc - Poac              #dc power wasted in transistor collector (in watt) 
no = Poac / Pindc                #Overall efficiency
nc = Poac / Ptrdc                #Collector efficiency  

#Result

print "Power supplied by the d.c. source to the amplifier circuit is ",Pindc," W."
print "D.C. power consumed by the load resistor is ",PRLdc," W."
print "A.C. power developed across the load resistor is ",Poac," W."
print "D.C. power delivered to the transistor is ",Ptrdc," W."
print "D.C. power wasted in the transistor collector is ",Pcdc," W."
print "Overall efficiency is ",no,"."
print "Collector efficiency is ",round(nc * 100,1),"% ."

import math

#Variables 

a = 15.0                     #Turns ratio
RL = 8.0                     #Load resistance (in ohm)   

#Calculation

R1L = a**2 * RL              #Effective resistance (in ohm)   

#Result

print "The effective resistance is ",R1L * 10**-3," kilo-ohm."

import math

#Variables 

RL = 16.0                    #Load resistance (in ohm)
R1L = 10.0 * 10**3           #Effective resistance (in ohm)

#Calculation

a = (R1L / RL)**0.5          #Turns ratio

#Result

print "Turns ratio is ",a,": 1."

import math

#Variables 

RL = 8.0                    #Load resistance (in ohm)
a = 10.0                    #Turns ratio
ICQ = 500.0 * 10**-3        #Collector current (in Ampere)

#Calculation

R1L = a**2 * RL             #Effective load (in ohm)
Poac = 1.0/2* ICQ**2 * R1L  #Maximum power delieverd (in watt)

#Result

print "The maximum power delievered to load is ",Poac," W."

import math

#Variables 

Ptrdc = 100.0 * 10**-3           #Maximum collector dissipated power (in watt)
VCC = 10.0                       #Source voltage (in volts)
RL = 16.0                        #Load resistance (in ohm)
no = nc = 0.5                    #Efiiciency

#Calculation

Poac = no * Ptrdc                #Maximum undistorted a.c. output power (in watt)
ICQ = 2 * Poac / VCC             #Quiescent collector current (in Ampere)
R1L = VCC / ICQ                  #Effective load resistance (in ohm)
a = (R1L / RL)**0.5

#Result

print "Maximum undistorted a.c. output power is ",Poac," W.\nQuiescent collector current is ",ICQ," A.\nTransformer turns ratio is ",round(a),"."

import math

#Variables 

VCC = 10.0                   #Source voltage (in volts)
Ip = 50.0 * 10**-3           #Collector current (in Ampere)
RL = 4.0                     #Load resistance (in ohm)     

#Calculation

I = round(Ip / 2**0.5,3)     #r.m.s. value of collector current (in Ampere)
Poac = I**2 * RL             #Average power delievered (in watt)
V1 = VCC                     #Primary voltage (in volts)
R1L = V1 / Ip                #Effective load resistance (in ohm)
a = round((R1L / RL)**0.5)   #Turns ratio
V2 = V1 / a                  #Secondary voltage (in volts)
I2p = V2 / RL                #Peak value of secondary current (in Ampere)
I2 = I2p / 2**0.5            #r.m.s. value of secondary current (in Ampere)
Pavg = I2**2 * RL            #Average power transferred to speaker (in watt) 

#Result

print "Power transferred when directly connected is ",Poac * 10**3," mW." 
print "The average power transferred to the speaker is ",round(Pavg,2) * 10**3," mW."

import math

#Variables 

RL = 1.0 * 10**3                   #Load resistance (in ohm)
IC = 10.0 * 10**-3                 #Collector current (in Ampere)

#Calculation

PL = IC**2 * RL                    #Load power (in watt)

#Result

print "Power delivered to the load is ",PL," W."

import math

#Variables 

RL = 8.0                   #Load resistance (in ohm)
VP = 16.0                  #Peak output voltage (in volts)

#Calculation

P = VP**2 / (2 * RL)       #Power drawn from the source (in watt)  

#Result

print "The power drawn from the source is ",P," W."

import math

#Variables 

Pcdc = 10.0              #Power rating of amplifier (in watt)
n = 0.785                #Maximum overall efficiency           

#Calculation

PT = 2 * Pcdc            #Total power dissipation of two transistors (in watt)
Poac = (PT * n) / (1-n)  #Maximum power output (in watt)

#Result

print "Maximum power output is ",round(Poac,2)," W."

import math

#Variables 

no = 0.6                  #efficiency 
Pcdc = 2.5                #Maximum collector dissipation of each transistor (in watt)

#Calculation

PT = 2 * Pcdc            #Total power dissipation of two transistors (in watt)
Pindc = PT / (1 - no )   #dc input power (in watt)
Poac = no * Pindc        #ac output power (in watt)     

#Result

print "The d.c. input power is ",Pindc," W.\nThe a.c. output power is ",Poac," W."

