from math import pi,sqrt
import math 
Vpiv=1500                   # peak inverse voltage
V=415                       # main supply
Vf=Vpiv/(sqrt(2)*V)         # voltage safety factor
Vf=round(Vf,2)
print  'value of voltage safety factor=',Vf

from math import pi,sqrt
import math 
Vf=2.1                    # voltage safety factor 
V=230                     # main supply
Vpiv=sqrt(2)*Vf*V         # peak inverse voltage
Vpiv=round(Vpiv,2)
print 'value of peak inverse voltage=',Vpiv,'volts'

import math 
C=30*10**-12                       # equivalent capacitance 
diffV=150*10**6                    # dv/dt value of capacitor
Ic=C*(diffV)                       # capacitive current
print 'value of capacitive current=',Ic,'Amp'

import math 
Ic=5.0                      # capacitive current in milli amperes
difV=175.0                  # dv/dt value in mega V/s
C=Ic/(difV)*10**3         # equivalent capacitance in pico farad
C=round(C,2)
print 'value of equivalent capacitance=',C,'pico farad'

import math 
Ic=6*10**-3             # capacitive current
C=25*10**-12            # equivalent capacitance
diffV=Ic/C              # dv/dt value of capacitor
print 'value of dv/dt=',diffV,'v/s'

import math 
Ic=5              # capacitive current in milli amperes
C=35              # equivalent capacitance in pico farad
difV=Ic*10**3/C   # value of dv/dt that can trigger the device in V/ microseconds
print  'value of dv/dt that can trigger the device=',difV,'V/microseconds'

from math import sqrt
import math 
Vpiv=1350            # peak inverse voltage in volts
V=415                # main supply in volts
Vf=Vpiv/(sqrt(2)*V)  # voltage safety factor
Vf=round(Vf,2)
print  'value of voltage safety factor=',Vf,'v'

