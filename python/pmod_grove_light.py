from pynq import Overlay
Overlay("base.bit").download()

from pynq.iop import Grove_Light
from pynq.iop import PMODA
from pynq.iop import PMOD_GROVE_G4

lgt = Grove_Light(PMODA, PMOD_GROVE_G4)

sensor_val = lgt.read()
print(sensor_val)

import time
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

lgt.set_log_interval_ms(100)
lgt.start_log()
# Change input during this time
time.sleep(10)
r_log = lgt.get_log()

plt.plot(range(len(r_log)), r_log, 'ro')
plt.title('Grove Light Plot')
min_r_log = min(r_log)
max_r_log = max(r_log)
plt.axis([0, len(r_log), min_r_log, max_r_log])
plt.show()

