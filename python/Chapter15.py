from numpy import sin, pi
# A sine wave of voltage varies from zero to a maximum of 100 V. How much is the voltage at the instant of 30° of the cycle? 45°? 90°? 270°?

# Given data

Vm = 100#       # Vm=100 Volts
t1 = 30#        # Theta 1=30°.
t2 = 45#        # Theta 2=45°.
t3 = 90#        # Theta 3=90°.
t4 = 270#       # Theta 4=270°.

v1 = Vm*sin(t1*pi/180)
print 'The Voltage at 30° = %0.2f Volts'%v1

v2 = Vm*sin(t2*pi/180)
print 'The Voltage at 45° = %0.2f Volts'%v2

v3 = Vm*sin(t3*pi/180)
print 'The Voltage at 90° = %0.2f Volts'%v3

v4 = Vm*sin(t4*pi/180)
print 'The Voltage at 270° = %0.2f Volts'%v4

from __future__ import division
# An alternating current varies through one complete cycle in 1 ⁄ 1000 s. Calculate the period and frequency.

# Given data

tc = 1/1000#        # One Complete Cycle=1 ⁄ 1000 sec.

T = tc#
print 'The Time period = %0.e Seconds'%T
print 'i.e 1/1000 sec'

f = 1/tc#
print 'The Frequency = %0.2f Hertz'%f
print 'OR 1 kHz'

# Calculate the period for the two frequencies of 1 MHz and 2 MHz.Calculate the period for the two frequencies of 1 MHz and 2 MHz.

# Given data

f1 = 1*10**6#        # Freq=1 MHz
f2 = 2*10**6#        # Freq=2 MHz

t1 = 1/f1#
print 'The Time period = %0.e Seconds of 1 MHz freq.'%t1
print 'i.e 1*10**-6 sec = 1 usec'

t2 = 1/f2#
print 'The Time period = %0.e Seconds of 2 MHz freq.'%t2
print 'i.e 0.5*10**-6 sec = 0.5 usec'

# Calculate lamda for a radio wave witf f of 30 GHz.

# Given data

c = 3*10**10#     # Speed of light=3*10**10 cm/s
f = 30*10**9#        # Freq=30 GHz

l = c/f#
print 'The Lamda or Wavelenght = %0.2f cm'%l

# The length of a TV antenna is lamda/2 for radio waves with f of 60 MHz. What is the antenna length in centimeters and feet?

# Given data

c = 3*10**10#     # Speed of light=3*10**10 cm/s
f = 60*10**6#     # Freq=60 MHz
In = 2.54#       # 2.54 cm = 1 in
ft = 12#         # 12 in = 1 ft

l1 = c/f#
l = l1/2#
print 'The Height = %0.2f cm'%l

li = l/In
lf = li/ft#
print 'The Height = %0.2f feet'%lf

# For the 6-m band used in amateur radio, what is the corresponding frequency?

# Given data

v = 3*10**10#     # Speed of light=3*10**10 cm/s
l = 6*10**2#      # lamda=6 meter

f = v/l
print 'The Frequency = %0.f Hertz'%f
print 'i.e 50*10**6 Hz OR 50 MHz'

# What is the wavelength of the sound waves produced by a loudspeaker at a frequency of 100 Hz?

# Given data

c = 1130#   # Speed of light=1130 ft/s
f = 100#    # Freq=100 Hz

l = c/f#
print 'The Lamda or Wavelenght = %0.2f ft'%l

# For ultrasonic waves at a frequency of 34.44 kHz, calculate the wavelength in feet and in centimeters.

# Given data

c = 1130#   # Speed of light=1130 ft/s
f = 34.44*10**3#    # Freq=100 Hz
In = 2.54#       # 2.54 cm = 1 in
ft = 12#         # 12 in = 1 ft

l = c/f#
print 'The Lamda or Wavelength = %0.2f ft'%l

a = l*ft#

l1 = a*In#
print 'The Lamda or Wavelength = %0.2f cm'%l1

