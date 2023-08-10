import math

#Variables

I0 = 2 * 10**-7              #Current (in Ampere)
VF = 0.1                     #Forward voltage (in volts)

#Calculation

I = I0 * (math.exp(40*VF)-1)      #Current through diode (in Ampere)

#Result

print "Current throrough diode is ",round(I*10**6,2)," micro-Ampere."

import math

#Variables

VF = 0.22                      #Forward voltage (in volts)
T = 298.0                      #Temperature (in kelvin)
I0 = 10**-3                    #Current (in Ampere)
n = 1

#Calculation

VT = T/11600                   #Volt equivalent of temperature (in volts)
I = I0*(exp(VF/(n*VT))-1)      #Diode Current (in Ampere) 

#Result

print "Diode current is ",round(I,1)," A."

import math

#Variables

I1 = 0.5 * 10**-3          #Diode current1 (in Ampere)
V1 = 340 * 10**-3          #Voltage1 (in volts)
I2 = 15 * 10**-3           #Diode current2 (in Ampere)
V2 = 440 * 10**-3          #Voltage2 (in volts)

#Calculation

n = 4/math.log(30)                #By solving both the given equations

#Result

print "Value of n is ",round(n,2),"."

import math

#Variables

I300 = 10 * 10**-6             #Current at 300 kelvin (in Ampere)
T1 = 300                       #Temperature (in kelvin)
T2 = 400                       #Temperature (in kelvin)

#Calculation

I400 = I300 * 2**((T2-T1)/10)  #Current at 400 kelvin (in Ampere)  

#Result

print "Current at 400 k is ",round(I400*10**3,1)," mA."

import math

#Variables

rb = 2               #bulk resistance (in ohm)
IF = 12 * 10**-3     #FOrward current (in Ampere)

#Calculation

VF = 0.6 + IF * rb   #Voltage drop (in volts)

#Result

print "Voltage drop across a silicon diode is ",VF," V."

import math

#Variables

T = 398.0                    #Temperature (in kelvin)
I0 = 30 * 10**-6             #Reverse saturation current (in Ampere)
V = 0.2                      #Voltage (in volts)

#Calculation

VT = T/11600                 #Volt equivalent of temperature (in volts)
I = I0 * (math.exp(V/VT)-1)       #Diode current (in Ampere)
rac = VT/I0 * math.exp(-V/VT)     #dynamic resistance in forward direction (in ohm)
rac1 = VT/I0 * math.exp(V/VT)     #dynamic resistance in reverse direction (in ohm)

#Result

print "Dynamic resistance in forward direction is ",round(rac,2)," ohm.\nDynamic resistance in backward direction is ",round(rac1/10**6,3)," Mega-ohm."

import math

#Variables

PDmax = 0.5              #power dissipation (in watt)
VF = 1                   #Forward voltage (in volts)
VBR = 150                #Breakdown voltage (in volts)

#Calculation

IFmax = PDmax/VF         #Maximum forward current (in Ampere)
IR = PDmax/VBR           #Breakdwon current that burns out the diode (in Ampere)

#Result

print "Maximum forward current is ",IFmax," A.\nBreakdwon current that burns out the diode is ",round(IR*10**3,2)," mA."

import math

#Variables

R = 330                   #Resistance (in ohm)
VS = 5                    #Source voltage (in volts)

#Calculation

VD = VS                   #Voltage drop across diode (in volts)
VR = 0                    #Voltage drop across the resistance (in volts)
I = 0                     #Current through circuit

#Result

print "Voltage drop across the diode is ",VD," V.\nVoltage drop across the resistance is ",VR," V.\nCurrent through the circuit is ",I," A."

import math

#Variables

VS = 12.0                     #Source coltage (in volts)
R = 470.0                     #Resistance (in ohm)

#Calculation

VD = 0                      #Voltage drop across diode (in volts)
VR = VS                     #Value of VR (in volts)
I = VS/R                    #Current (in Ampere)

#Result

print "Value of VD is ",VD," V.\nValue of VR is ",VR," V.\nCurrent through the circuit is ",round(I*10**3,2)," mA."

import math

#Variables

VS = 6                 #Source voltage (in volts)
R1 = 330               #Resistance (in ohm)
R2 = 470               #Resistance (in ohm)
VD = 0.7               #Diode voltage (in volts)

#Calculation

RT = R1 + R2           #Total Resistance (in ohm)
I = (VS - 0.7)/RT      #Current through the diode

#Result

print "Current through the circuit is ",I * 10**3," mA."

import math

#Variables

VS = 5                 #Source voltage (in volts)
R = 510                #Resistance (in ohm)
VF = 0.7               #Forward voltage drop (in volts)

#Calculation

VR = VS - VF           #Net voltage (in volts)
I = VR / R             #Current through the diode

#Result

print "Voltage across the resistor is ",VR," V.\nThe circuit current is ",round(I * 10**3,2)," mA."

import math

#Variables

VS = 6                         #Source voltage (in volts)
VD1 = VD2 = 0.7                #Diode Voltage drop (in volts)
R = 1.5 * 10**3                #Resistance (in ohm)

#Calculation

I = (VS - VD1 - VD2)/R         #Current (in Ampere)

#Result

print "Total current through the circuit is ",round(I * 10**3,3)," mA." 

import math

#Variables

VS = 12                          #Source voltage (in volts)
R1 = 1.5 * 10**3                 #Resistance (in ohm)
R2 = 1.8 * 10**3                 #Resistance (in ohm)
VD1 = VD2 = 0.7                  #Diode Voltage drop (in volts)

#Calculation

RT = R1 + R2                     #Total Resistance (in ohm)
I = (VS - VD1 - VD2)/RT          #Current (in Ampere)

#Result

print "Total current through the circuit is ",round(I * 10**3,3)," mA." 

import math

#Variables

R = 3.3 * 10**3              #Resitance (in ohm) 

#Calculation

#Case (a)

V11 = V21 = 0                #Voltages (in volts)
V01 = 0                      #Output Voltage (in volts)

#Case (b)

V21 = 0                      #Voltage (in volts)
V22 = 5                      #Voltage (in volts)
V02 = V22 - 0.7              #Output voltage (in volts)  

#Case (c)

V31 = 5                      #Voltage (in volts)
V32 = 0                      #Voltages (in volts)
V03 = V31 - 0.7              #Output voltage (in volts)  

#Case (d)

V41 = V42 = 5                #Voltages (in volts)
V04 = V41 - 0.7              #Output voltage (in volts)  

#Result

print "Output Voltage in case 1 is ",V01," V.\nOutput Voltage in case 2 is ",V02," V.\nOutput Voltage in case 3 is ",V03," V.\nOutput Voltage in case 4 is ",V04," V."

import math
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import plot,title,xlabel,ylabel

#Variables

rB = 1.0                                    #bulk resistance (in ohm)
V = 10 * 10**-3                             #Signal Amplitude (in volts)

#Calculation

#Case (a)

R = 20.0                                    #Resitance (in kilo-ohm)
Vg = 20.0                                   #Source voltage (in volts)
I = (Vg - 0.7)/R                            #Current (in milli-Ampere)         

#Case (b)

rj = 50.0                                   #junction resistance (in ohm)
re = rB + rj                                #a.c. resistance (n ohm)
rnet = re * (R*10**3)/(re + (R*10**3))      #Net resistance (in ohm)
V1 = V * re/(re + 1000)                     #Voltage drop across 51 ohm resitance (in ohm)

#Result

print "Current in dc circuit is ",round(I)," mA.\na.c voltage drop across 51 ohm resistance is ",round(V1*10**3,3)," mV."

#Graph

x = numpy.linspace(-4*math.pi,4*math.pi,500)
y = numpy.sin(x)
plot(x,0.7 + 0.48*10**-3*y)
title("Total Voltage 'V' across the diode")
xlabel("t(in seconds)->")
ylabel("Voltage(in volts)->")

