from __future__ import division
import math

#variable declaration
E= 2            #electric field strength of wave in V/m
n=120*math.pi   #where n is mu [free space impedence (120xpi)]

#calculations
H=E/n           # As n = E/H
H=H*10**3

#results
print "strength of magnetic field H in free space is %r mA/metre" % round(H,3)

from __future__ import division
import math

#variable Declaration
P= 625*10**3        #power of transmitting antenna in Watt
r=30*10**3          #distance in meter

#calculations
Erms=math.sqrt(90*P)/r
Erms=Erms*10**3

#Results
print"Field strength is %r mV/metre." %round(Erms,3)

from __future__ import division
import math

#variable Declaration
f=10            #frequency in Mega Hertz
le=60           #Height of antenna in metres
lemda=300/f

#calculations
Rr= 160*(math.pi)**2*le**2/lemda**2
Rr=Rr/10**3

#Results
print "Radiation resistance of antenna is %r Kilo ohms." %round(Rr,3)

from __future__ import division
import math

#variable Declaration
le=100          # height of antenna in Metres
Irms=450        # current at the base in Amperes
f= 40000.0      # frequency in Hertz
f=f/10**6       # frequency in Mega Hertz

lemda= 300/f    # as per the formula where frequncy in MHz

#calculations
Rr=160*(math.pi)**2*le**2/lemda**2 #Rr is radiated resistance in ohms

#Results
print "Radiation resistance is %r ohms."%round(Rr,2)

#calculations
Pr= Irms**2*Rr  # Power radiated in Watts
Pr= Pr/10**3    # Power radiated in Kilo Watts

#Results
print "Power radiated is %r kW."%round(Pr,2)

#variable Declaration
Rl=1.12         # otal reistance of antenna circuit in ohms
n= Rr/Rl        # Efficiancy of the antenna n= Radiation Resistance/Total antenna Resistance

#calculations
nper= n*100     # Efficiancy of the antenna (in percentage)

#Results
print "Efficiancy of antenna is %r percent. "%round(nper,2)


#variable Declaration
I=20            # current in Amperers
Rrad= 50        # Radiated resistance in Ohms

#calculations
Pr= I**2*Rrad   # Power radiated in watts

#Results
print "Antenna will radiate %r Watts of power."%Pr


#variable Declaration
P=5*10**3         # Power radiated in watts
I= 15.0           # current in Ampers
#calculations
Rrad=P/I**2       # Radiated resistance in Ohms

#Results
print "Radiated power is %r ohms."%round(Rrad,2)

from __future__ import division
import math

#variable Declaration
Rrad= 75        # Radiation resistance in ohms
Pr= 10          # Power radiated in kW
Pr=Pr*10**3     # Power radiated in W

#calculations
I=math.sqrt(Pr/Rrad)

#Results
print "%r Amperes current flows in the antenna"%round(I,2)

from __future__ import division
import math

#variable Declaration
P=W= 100*10**3        # power radiated in watt
r= 100              # distance in kilo metres
r=r*10**3           # distance in metres

#calculations
Erms= math.sqrt(90*W)/r

#Results
print "Strength of electric field Erms is %r V/m."%Erms

from __future__ import division
import math

#variable Declaration
Irms= 25        # transmitting antenna rms current in A
f=0.150         # frequency in Mega Hertz(MHz)
Erms= 1.5       # Field strength in mV/m
Erms=Erms/10**3 # Field strength in V/m
r=25            # distance in kilo metre
r=r*10**3       # distance in metre

#calculations
lemda= 300/f
le= Erms*lemda*r/(60*math.pi*Irms) # le is effective height of the antenna in metres

#Results
print "Effective heigth of the antenna le = %r metres."%round(le,2)

from __future__ import division
import math

le= 100           # Heigth of the antenna in metre
Irms= 100         # rms current in amperes
r=10             # distance in kilo metre
r=r*10**3           # distance in metre
f=300.0             # frequency in KHz
f=f/10**3           # frequency in MHz

lemda=300/f

#Calculations
Erms= (120*math.pi*Irms*le)/(lemda*r)

Rr= (160*(math.pi)**2*le**2)/(lemda**2)

P= Irms**2*Rr       # Power radiated in watts
P= P/10**3          # POwer radiated in kilo watts(kW)

#Results
print "(i)Field strength at Erms is %r mV/m"%round(Erms*10**3,2)
print "(ii)The power radiated is %r kW." %round(P,2)

