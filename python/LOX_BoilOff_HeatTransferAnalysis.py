# Import packages here:

import math as m
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt

# Properties of Materials (engineeringtoolbox.com, Cengel, Tian)

# Conductivity
Kair = 0.026  # w/mk
Kptfe = 0.25  # w/mk
Kcf = 0.8  # transverse conductivity 0.5 -0.8 w/mk

# Fluid Properties

rhoLox = 1141  # kg/m^3
TLox = -183  # *C

# Latent Heat of Evaporation
heOxy = 214000  # j/kg


# Layer Dimensions:

r1 = 0.0381  # meters (1.5")
r2 = 0.0396  # m
r3 = 0.0399  # m
r4 = 0.0446  # m
r5 = 0.0449  # m
L = 0.13081  # meters (5.15")

# Environmental Properties:

Ts = 38  # *C
T1 = -183  #*C

Rptfe = m.log(r2/r1)/(2*m.pi*L*Kptfe)
Rcf1 = m.log(r3/r2)/(2*m.pi*L*Kcf)
Rair = m.log(r4/r3)/(2*m.pi*L*Kair)
Rcf2 = m.log(r5/r4)/(2*m.pi*L*Kcf)

Rtot = Rptfe + Rcf1 + Rair + Rcf2 

print('Total Thermal Resistance equals: ', "%.2f" % Rtot, 'K/W')

#Heat transfer rate: 
Qrate = (Ts - T1)/Rtot

print('Calculated Heat Transfer rate equals: ',"%.2f" % Qrate, 'W')

EvapRate = Qrate/heOxy
print ('The rate of evaporation is', "%.6f" % EvapRate, 'kg/s')

VLox = m.pi*(r1)**2*L
mLox = rhoLox*VLox
print('The mass of the liquid oxygen in tank is: ', "%.2f" % mLox, 'kg')

Tboiloff = mLox/EvapRate/60
print('%.2f' % Tboiloff, 'minutes' )



