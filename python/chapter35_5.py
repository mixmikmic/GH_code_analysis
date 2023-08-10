import math

#Variables

Adm = 200000.0                    #Differential gain
Acm = 6.33                        #Common mode gain                    

#Calculation

CMRR = 20 * math.log10(Adm / Acm)      #Common-mode rejection ratio (in Decibels)   

#Result

print "The common-mode rejection ratio is ",round(CMRR)," dB."

import math

#Variables

CMRR = 90.0                    #Common-mode rejection ratio (in Decibels)
Adm = 30000.0                  #Differential gain

#Calculation

Acm = 10**(-CMRR/20.0) * Adm   #Common-mode gain 

#Result

print "The common-mode gain is ",round(Acm,3),"."

import math

#Variables

Slew_rate = 0.5 * 10**6                  #Slew rate (in volt per second)
Vpk = 100.0 * 10**-3                     #Peak-to-peak voltage (in volts)

#Calculation

fmax = Slew_rate / (2 * math.pi * Vpk)   #Maximum operating frequency (in Hertz)

#Result

print "The maximum operating frequency for the amplifier is ",round(fmax * 10**-3)," kHz."

import math

#Variables

Slew_rate1 = 0.5 * 10**6                  #Slew rate (in volt per second)
Slew_rate2 = 13.0 * 10**6                 #Slew rate (in volt per second)
Vpk = 10.0                                #Peak-to-peak voltage (in volts)

#Calculation

fmax = Slew_rate1 / (2 * math.pi * Vpk)   #Maximum operating frequency1 (in Hertz)
fmax1 = Slew_rate2 / (2 * math.pi * Vpk)  #Maximum operating frequency2 (in Hertz)

#Result

print "The maximum operating frequency for TLO 741 is ",round(fmax * 10**-3,3)," kHz.\nThe maximum opearing frequency for TLO 81 is ",round(fmax1 * 10**-3,1)," kHz."

#Slight variation due to higher precision.

import math

#Variables

ACL = 200.0                  #Closed loop voltage gain
Vout = 8.0                   #Output voltage (in volts)

#Calculation

Vin = - Vout / ACL           #Input a.c. voltage (in volts)

#Result

print "Maximum allowable input voltage (Vin) is ",abs(Vin * 10**3)," mV." 

import math

#Variables

ACL = 150.0                  #Closed loop voltage gain
Vin = 200.0 * 10**-3         #Input a.c. voltage (in volts) 
V = 12.0                     #Voltage (in volts)       

#Calculation

Vout = ACL * Vin             #Output voltage (in volts)
Vpkplus = V -2.0             #maximum positive peak voltage (in volts)
Vpkneg = -V + 2.0            #maximum negative peagk voltage (in volts) 

#Result

print "The maximum possible output value could be between ",Vpkplus," V and ",Vpkneg," V."

import math

#Variables

R1 = 1.0 * 10**3               #Resistance (in volts)
R2 = 10.0 * 10**3              #Resistance (in volts)
vinmin = 0.1                   #Input voltage minimum (in volts)
vinmax = 0.4                   #Input voltage maximum (in volts)

#Calculation

ACL = R2 / R1                  #Closed loop voltage gain
Voutmin = ACL * vinmin         #Minimum output voltage (in volts)
Voutmax = ACL * vinmax         #Maximum output voltage (in volts)

#Result

print "The value of output voltage increases from ",Voutmin," V to ",Voutmax," V."

import math

#Variables

R1 = 1.0 * 10**3                   #Resistance (in ohm)
R2 = 2.0 * 10**3                   #Resistance (in ohm)
V1 = 1.0                           #Voltage (in volts)

#Calculation

ACL = R2 / R1                      #Closed loop voltage gain   
vo = ACL * V1                      #Output voltage (in volts)

#Result

print "Output voltage of the inverting amplifier is ",vo," V."

import math

#Variables

R2 = 100.0 * 10**3                 #Resistance (in ohm)
R1 = 10.0 * 10**3                  #Resistance (in ohm)  
ACM = 0.001                        #Common-mode gain 
Slew_rate = 0.5 * 10**6            #Slew rate (in volt per second)   
Vpk = 5.0                          #Peak voltage (in volts)

#Calculation

ACL = R2 / R1                       #Closed loop voltage gain
Zin = R1                            #Input impedance of the circuit (in ohm)
Zout = 80.0                         #Output impedance of the circuit (in ohm)
CMRR = ACL / ACM                    #Common mode rejection ratio    
fmax = Slew_rate / (2*math.pi*Vpk)  #Maximum frequency (in Hertz)

#Result

print "Closed-loop gain is ",ACL,".\nInput impedance is ",Zin * 10**-3," kilo-ohm.\nOutput impedance is ",Zout," ohm.\nCommon-mode rejection ratio is ",CMRR,".\nMaximum operating frequency is ",round(fmax * 10**-3,1)," kHz."

import math

#Variables

R2 = 100.0 * 10**3                 #Resistance (in ohm)
R1 = 10.0 * 10**3                  #Resistance (in ohm)  
Slew_rate = 0.5 * 10**6            #Slew rate (in volt per second)   
Vpk = 5.5                          #Peak voltage (in volts)
RL = 10.0 * 10**3                  #Load resistance (in ohm)  
ACM = 0.001                        #Common mode gain 

#Calculation

ACL = (1 + R2/R1)                  #Closed loop voltage gain     
CMRR = ACL / ACM                   #Common-mode rejection ratio       
vin = 1.0                          #Voltage (in volts)
Vout = ACL * vin                   #Output voltage (in volts)
Vpk = 5.5                          #Peak-to-peak voltage (in volts)    
fmax = Slew_rate/(2*math.pi*Vpk)   #Maximum frequency (in Hertz)

#Result

print "Closed loop gain is ",ACL,".\nCMRR is ",CMRR,".\nMaximum operating frequency is ",round(fmax * 10**-3,2)," kHz."

import math

#Variables

ACL = 1.0                          #Closed loop gain
Acm = 0.001                        #Common mode gain      
Slew_rate = 0.5 * 10**6            #Slew rate (in Volt per second)

#Calculation

CMRR = ACL / Acm                   #Common-mode rejection ratio       
vin = 1.0                          #Voltage (in volts)
Vout = ACL * vin                   #Output voltage (in volts)
Vpk = 3.0                          #Peak-to-peak voltage (in volts)    
fmax = Slew_rate/(2*math.pi*Vpk)   #Maximum frequency (in Hertz)

#Result

print "ACL is ",ACL,".\nCMRR is ",CMRR,".\nfmax is ",round(fmax * 10**-3,1)," kHz."

import math

#Variables

V1 = 0.1                                          #Voltage (in volts)
V2 = 1.0                                          #Voltage (in volts)
V3 = 0.5                                          #Voltage (in volts)         
R1 = 10.0 * 10**3                                 #Resistance (in ohm)
R2 = 10.0 * 10**3                                 #Resistance (in ohm)
R3 = 10.0 * 10**3                                 #Resistance (in ohm)
R4 = 22.0 * 10**3                                 #Resistance (in ohm)

#Calculation

Vout = (-R4/R1*V1) + (-R4/R2*V2) + (-R4/R3*V3)    #Output voltage (in volts)

#Result

print "Output voltage is ",abs(Vout)," V."

import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,ylabel,xlabel,title

#Variables

#V1 = 2 * sin(wt)                                 #Voltage (in volts)
V2 = 5.0                                          #Voltage (in volts)
V3 = -100.0 * 10**-3                              #Voltage (in volts) 

#Result

#Vo = 4 * V1 + V2 + 0.1 * V3                      #Output voltage

#Graph

x = numpy.linspace(0,10,200)
y = numpy.sin(x)
plot(x,-15 + 8*y)
plot(x,x-x-15,'')
title("output waveform")
ylabel("output voltage (Vo)")
xlabel("t")

import math

#Variables

V1 = -2.0                                          #Voltage (in volts)
V2 = 2.0                                           #Voltage (in volts)
V3 = -1.0                                          #Voltage (in volts)         
R1 = 200.0 * 10**3                                 #Resistance (in ohm)
R2 = 250.0 * 10**3                                 #Resistance (in ohm)
R3 = 500.0 * 10**3                                 #Resistance (in ohm)
Rf = 1.0 * 10**6                                   #Resistance (in ohm)

#Calculation

Vout = (-Rf/R1*V1) + (-Rf/R2*V2) + (-Rf/R3*V3)     #Output voltage (in volts)

#Result

print "Output voltage is ",Vout," V."

