from pynq import Overlay
Overlay("base.bit").download()

from pynq.iop import Grove_ADC
from pynq.iop import PMODA
from pynq.iop import PMOD_GROVE_G4 

grove_adc = Grove_ADC(PMODA, PMOD_GROVE_G4)
print("{} V".format(round(grove_adc.read(),4)))

grove_adc.set_log_interval_ms(100)
grove_adc.start_log()

log = grove_adc.get_log()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.plot(range(len(log)), log, 'ro')
plt.title('Grove ADC Voltage Log')
plt.axis([0, len(log), min(log), max(log)])
plt.show()

from pynq.iop import Grove_ADC
from pynq.iop import ARDUINO
from pynq.iop import ARDUINO_GROVE_I2C

grove_adc = Grove_ADC(ARDUINO, ARDUINO_GROVE_I2C)
print("{} V".format(round(grove_adc.read(),4)))

grove_adc.set_log_interval_ms(100)
grove_adc.start_log()

log = grove_adc.get_log()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.plot(range(len(log)), log, 'ro')
plt.title('Grove ADC Voltage Log')
plt.axis([0, len(log), min(log), max(log)])
plt.show()

