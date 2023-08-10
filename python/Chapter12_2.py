import math

#Variable Declaration

#For Scale reading =10 V, and precise voltage=9.5 V
scale_reading=10                                            #Scale reading is 10 V

precise_reading=9.5                                         #Precise voltage is 9.5 V

error=(precise_reading-scale_reading)/scale_reading*100     #Error in percentage form w.r.t reading

error_fullscale=(precise_reading-scale_reading)*100/100     #Error with respect to full scale 


print "When scale reading is 10 V and precise voltage is 9.5 V,"
print "Error=-",round(error,1),"% of reading=",error_fullscale, "% of full scale"

print 
#For Scale reading =50 V, and precise voltage=51.7 V
scale_reading=50                                            #Scale reading is 50 V
precise_reading=51.7                                        #Precise voltage is 51.7 V
error=(precise_reading-scale_reading)/scale_reading*100     #Error in percentage form 
error_fullscale=(precise_reading-scale_reading)*100/100

print "When scale reading is 50 V and precise voltage is 51.7 V,"
print "Error= +",round(error,1),"% of reading= +",error_fullscale, "% of full scale"

import math

#Variable Declaration

V=114               #Measured Voltage in V
I=1                 #Measured Current in A
W=120               #Full Scale wattage in W

P=V*I               #Wattmeter Power
error=P-W           #Correction figure
print "Correction figure=",error,"W"

error=error*100/W   #Error %

print "Error=",error,"%"

import math

#Variable Declaration

R4=1125.0
R5=4017.9
Vz=6.4
accuracy=100.0/10**6           #100ppm

#Calculation
#Maximum and Minimum values of resistances in ohm
R4max=R4*(1+accuracy)        
R4min=R4*(1-accuracy)
R5max=R5*(1+accuracy)
R5min=R5*(1-accuracy)

#Maximum and minimum zener voltages in V
Vzmax=Vz+Vz*0.01/100                  #Maximum voltage is Vz+0.01% of Vz
Vzmin=Vz-Vz*0.01/100                  #Minimum voltage is Vz-0.01% of Vz

#Maximum and minimum output voltages in V
Vomax=Vzmax*(R5max/(R4min+R5max))     #Output is maximum when Vz is maximum, R5 is minimum and R4 is maximum
Vomin=Vzmin*(R5min/(R4max+R5min))     #Output is minimum when Vzi mimimum, R5 is maximum and R4 is minimum
Vo=Vz*(R5/(R4+R5))

error=round(Vomax-Vo,4)               #Deviation of output voltage from theoretical value 

#Result
print "Therefore Vo=",int(Vo),"V ±",error*10**6,"micro volt"

import math

#Variable Declaration

Rab=100        #Resistance of wire AB, in ohm
Vb1=3          #Battery B1, terminal voltage(V)
Vb2=1.0190     #Standard Cell Voltage(V) 
l=50.95        #Length BC, in cm

#At Calibration

Vbc=Vb2                              
volt_per_unit_length=Vbc/l        #in V/cm
Vab=100*volt_per_unit_length      #in V 
I=Vab/Rab                         #Ohm's Law
Vr1=Vb1-Vab                       #KVL 
R1=Vr1/I                           

#At 94.3cm
Vx=94.3*volt_per_unit_length

#Worst case: Terminal voltage of B2 or B1 may be reversed
#Total voltage producing current flow through standard cell is

Vt=Vb2+Vb1
R2=Vt/(20*10**-6)       #Value of resistance R2 to limit standard cell current to a maximum of 20 micro ampere


print "When the potentiometer is calibrated, I=",I*10**3,"mA"
print "R1=",R1,"ohm"

print 
print "Vx=",round(Vx,3),"V"
print 
print "The value of R2 to limit standard cell current to 20 micro ampere is ",int(R2*10**-3),"kilo ohm"

import math

R3=509.5                   #in ohm
R4=290.5                   #in ohm
R13=100                    #in ohm
l=100                      #in cm
Vb2=1.0190                 #in V(Standard Cell Voltage)

Vr3=Vb2     
I1=Vb2/R3                  #Ohm's Law 
 
#Maximum measurable voltage:
Vae=I1*(R3+R4)             #Maximum measurable voltage in V

#Resolution
I2=Vae/(8*R13)             #in A     

Vab=I2*R13
slidewire_vper_length=Vab/l   #in V/mm

instrument_resolution=slidewire_vper_length*1  #As contact can be read within 1 mm, 1 is multiplied

print "The instrument can measure a maximum of",Vae,"V"
print "Instrument resolution=±",instrument_resolution*10**2,"mV"

