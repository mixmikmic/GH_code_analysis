from time import sleep
from pynq import Overlay
from pynq.iop import Pmod_ADC
from pynq.iop import PMODA
from pynq.iop import PMODB

ol = Overlay("base.bit")
ol.download()

if_id = input("Type in the interface ID used (PMODA or PMODB): ")
if if_id.upper()=='PMODA':
    adc = Pmod_ADC(PMODA)
else:
    adc = Pmod_ADC(PMODB)

freq = int(input("Type in the frequency/Hz of the waveform: "))
period = 1/freq
log_interval_us = 0

# Assume Channel 0 is connected to the waveform generator
adc.start_log(1,0,0,log_interval_us)
sleep(3*period)
log = adc.get_log()

# Draw the figure
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.plot(range(len(log)), log, 'ro')
plt.title('PMOD ADC Waveform')
plt.axis([0, len(log), min(log), max(log)])
plt.show()

adc.reset()
del adc,ol

