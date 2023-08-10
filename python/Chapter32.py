from math import log
# Calculate the frequency of the emmiter voltage waveform. Assume n=0.6

# Given data

Rt = 220*10**3#      # Resistor Rt=220k Ohms
Ct = 0.1*10**-6#     # Capacitor Ct=0.1u Farad
n = 0.6#            # Constant

A = 1./(1-n)#
T = Rt*Ct*log(A)#

f = 1./T#
print 'The Frequency of the Emmiter Voltage Waveform = %0.2f Hertz'%f

