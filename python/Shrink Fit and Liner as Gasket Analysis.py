# Import packages here:

import math as m
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt

# Properties of Materials (engineeringtoolbox.com, Cengel, Tian, DuPont, http://www.dtic.mil/dtic/tr/fulltext/u2/438718.pdf)

# Coefficient of Thermal Expansion
alphaAluminum = 0.0000131  # in/in/*F
alphaPTFE = 0.0000478  # in/in/*F (over the range in question)

# Elastic Moduli

EAluminum = 10000000  # psi
EAluminumCryo = 11000000  # psi
EPTFE = 40000  # psi
EPTFECryo = 500000  # psi

# Yield Strength
sigmaY_PTFE = 1300  # psi
sigmaY_PTFECryo = 19000  # psi

# Poisson's Ratio

nuAluminum = 0.33  # in/in
nuPTFE = 0.46  # in/in

# Temperature Change Between Ambient and LN2  

DeltaT = 389  # *F

# Geometry of Parts

# Main Ring Outer Radius
roMain = 2.0000  # in

# End Cap Inner Radius
riCap = 1.3750  # in

# Interfacial Radius
r = 1.5000  # in

# Liner Thickness
t = 0.125  # in

m = 2.00
P = 45  # psi
yAmbient = 1200  # psi
sigmaPTFEAmbient1 = yAmbient
sigmaPTFEAmbient2 = m*P
sigmaPTFEAmbient = sigmaPTFEAmbient1

deltaLinerAmbient = (sigmaPTFEAmbient/EPTFE)*t
print('The change in liner thickness due to compression must be', "%.4f" % deltaLinerAmbient, 'in, in order to achieve a proper seal.')

rCryo = r - r*alphaAluminum*DeltaT 
Deltar = r - rCryo

print('The maximum change in end cap radius equals: ', "%.4f" % DeltaR, 'in')
print('This means that the maximum theoretical interference for the shrink fit is ', "%.4f" % DeltaR, 'in')

deltaLinerAmbientMax = DeltaR - 0.00125

print('The achievable ambient temperature change in liner thickness due to shrink fitting is', "%.4f" % deltaLinerAmbientMax, 'in')

tCryo = t - t*alphaPTFE*DeltaT
print ('The liner thickness at cryogenic temperature is', "%.4f" % tCryo,'in')
deltat = t*alphaPTFE*DeltaT
print ('The change in liner thickness due to thermal contraction is', "%.4f" % deltat, 'in')
tGap = t - deltaLinerAmbient
print ('The ambient temperature liner gap width is', "%.4f" % tGap, 'in')
deltaGap = tGap*alphaAluminum*DeltaT
print ('The change in gap width is', "%.4f" % deltaGap, 'in')
deltaLinerCryo = deltaLinerAmbient + deltaGap - deltat
print ('The total change in liner thickness at cryogenic temperature is', "%.4f" % deltaLinerCryo, 'in')
sigmaPTFECryo = (deltaLinerCryo/tCryo)*EPTFECryo
print('Thus, the maximum achievable pressure exerted on the PTFE at cryogenic temperature is', "%.2f" % sigmaPTFECryo, 'psi')

h = 0.125
mu = 1.2
deltaInterference = ((2*P*r**4)/(mu*h*EAluminum))*((roMain**2 - riCap**2)/((roMain**2 - r**2)*(r**2 - riCap**2)))
print('The intereference thickness needed to overcome the pressure force on the end caps is', "%.4f" % deltaInterference, 'in')



