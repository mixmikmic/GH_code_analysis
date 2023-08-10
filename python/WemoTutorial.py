import pywemo
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

print("Current environment directory:" + sys.prefix)
print("System version: "+sys.version)
print("Current working directory: "+os.getcwd())

devices = pywemo.discover_devices()
devices

devices[0].Lights

lights = dict(zip(['bedroom','kitchen','bathroom'],devices[0].Lights.values()))
lights

lights['bedroom'].toggle()
time.sleep(3)
lights['bedroom'].toggle()

lights['bedroom'].turn_on(transition=2)
time.sleep(2.1)
lights['bedroom'].turn_off()

lights['bathroom'].turn_off()
lights['bathroom'].set_temperature(kelvin=2700, delay=False)
lights['bathroom'].turn_on()
time.sleep(0.5)
lights['bathroom'].set_temperature(kelvin=6500,transition=2,delay=False)
time.sleep(2.2)
lights['bathroom'].set_temperature(kelvin=2700,transition=2,delay=False)
time.sleep(2.2)

max_brightness = 255
x = np.arange(0,2,0.01)
brightness = np.cos(2*np.pi * x)*max_brightness/2. + max_brightness/2.
plt.plot(brightness)

for y in brightness:
    lights['bathroom'].turn_on(level=y,transition=0.1,force_update=True)
    time.sleep(0.11)

temp_range = 6500-2700
temp_base = 2700
temperature = np.cos(np.pi * x)*temp_range*0.5 + temp_range*0.5 + temp_base

for i in range(len(x)):
    lights['bathroom'].set_temperature(kelvin=temperature[i], delay=False)
    lights['bathroom'].turn_on(level=brightness[i],transition=0.1,force_update=True)

