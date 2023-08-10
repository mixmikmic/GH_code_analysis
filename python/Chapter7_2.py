import math

#Variable Declaration

I=0.5         #in A
E1=500        #E+Ea in V
Ra=10         #in ohm

#Calculations
R1=E1/I       #in ohm
R=R1-Ra       #in ohm

#Result
print "R=",int(R),"ohm"

import math

#Variable Declaration

sensitivity=10**3                             #in ohm/V
V=1000.0                                      #in V 
R=990.0                                       #in ohm
Ra=10.0                                       #in ohm
supply_voltage=500                            #in V 


#Calculations
Rv=V*sensitivity                              #in ohm
R1=Rv*R/(Rv+R)                                #in ohm 
voltmeter_reading=supply_voltage*R1/(Ra+R1)   #in volt  
ammeter_reading=supply_voltage/R1             #in A

#Results
print "Voltmeter Reading=",round(voltmeter_reading,1),"V"
print "Ammeter Reading=",round(ammeter_reading,1),"A"

import math


#For figure 7-1(a)
voltmeter_reading=495                 #in V
ammeter_reading=0.5                   #in A
R=voltmeter_reading/ammeter_reading   #in ohm
print "For V=495 V,I=0.5 A, R=",R,"ohm"

#For figure 7-1(b)
voltmeter_reading=500                 #in V
ammeter_reading=0.5                   #in A
R=voltmeter_reading/ammeter_reading   #in ohm

print "For V=500 V,I=0.5 A, R=",R,"ohm"

import math 

#Variable Declaration

#Bridge Resistances
P=3.5*10**3             #in ohm
Q=7*10**3               #in ohm
S=5.51*10**3            #in ohm 

#Calculations

R=S*P/Q                  #Equation for unknown resistance in a balanced bridge(ohm)

#When S=1 kilo ohm
S=1*10**3                #in ohm
R1=S*P/Q                 #in ohm  

#When S=8 kilo ohm
S=8*10**3                #in ohm 
R2=S*P/Q                 #in ohm

print "R=",round(R/1000,3),"kilo ohm"
print "Measurement Range is",round(R1),"ohm to ",round(R2/1000),"kilo ohm"

import math

#Variable Declaration
#Bridge Resistances
P=3.5*10**3             #in ohm
Q=7*10**3               #in ohm
S=5.51*10**3            #in ohm 
R=2.755*10**3           #in ohm 
p_accuracy=0.05         #in percentage 
q_accuracy=0.05         #in percentage
s_accuracy=0.1          #in percentage 

#Calculation
error_r=p_accuracy+q_accuracy+s_accuracy     #in percentage
Rmax=R+R*error_r/100.0                       #in ohm
Rmin=R-R*error_r/100.0                       #in ohm  

#Result

print "Error in R=±",round(error_r,1),"%"
print "R=",round(R/1000,3),"kilo ohm ±",round(error_r,1),"%"
print "R=",round(R/1000,3),"kilo ohm ±",round(R*error_r/100.0,1),"%"
print "R=",round(Rmin/1000,4),"kilo ohm to ",round(Rmax/1000,4),"kilo ohm"

import math

#Variable Declaration

P=3.5*10**3                  #in ohm
Q=7*10**3                    #in ohm
S=4*10**3                    #in ohm
R=2*10**3                    #in ohm
Eb=10
Ig=10**-6                    #in A/mm
Rg=2.5*10**3                  #in ohm

#Calculations
r=P*R/(P+R)+Q*S/(Q+S)        #R=P||R+Q||S in ohm
dV=Ig*(r+Rg)                 # Smallest voltage change in V  

Vr=Eb*R/(P+R)                #Voltage across R(Voltage Divider Rule), in V 
V=Vr+dV                      #in V   
Vp=Eb-V                      #KVL                  
Ip=Vp/P                      #Ohm's Law
Ir=Ip                
dR=round(V,5)/round(Ir,6)-R  #in ohm


print "Minimum Change in R is",round(dR,1),"ohm"

import math


#Variable Declaration
S=0.10       #in ohm
Q=0.15       #in ohm(Approximately equal to 0.15)

#Result
print "R/P=S/Q= ",int(S*100),"/",int(Q*100)

import math

#Variable Declaration
E=10000                     #in Volt
Iv=1.5*10**-6               #in A
rv=E/Iv                     #Volume resistance in ohm 

#Surface leakage Resistance

It=5*10**-6                 #in A
Is=It-Iv                    #KCL 
rs=E/Is                     #Surface Resistance in ohm

#Results
print "Volume resistance=",'%.1e' %rv,"ohm"
print "Surface resistance=",'%.1e' %rs,"ohm"

