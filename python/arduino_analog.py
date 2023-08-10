# Make sure the base overlay is loaded
from pynq import Overlay
Overlay("base.bit").download()

from pynq.iop import Arduino_Analog
from pynq.iop import ARDUINO
from pynq.iop import ARDUINO_GROVE_A1
from pynq.iop import ARDUINO_GROVE_A4

analog1 = Arduino_Analog(ARDUINO,ARDUINO_GROVE_A1)

analog1.read()

analog1.read_raw()[0]

from time import sleep

analog1.set_log_interval_ms(100)
analog1.start_log()

log1 = analog1.get_log()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(range(len(log1[0])), log1[0], 
                  'ro', label="X-axis of joystick")
line2, = plt.plot(range(len(log1[1])), log1[1], 
                  'bs', label="Y-axis of joystick")
plt.title('Arduino Analog Voltage Log')
plt.axis([0, len(log1[0]), 0.0, 3.3])
plt.legend(loc=4,bbox_to_anchor=(1, -0.3),
           ncol=2, borderaxespad=0.,
           handler_map={line1: HandlerLine2D(numpoints=1),
                        line2: HandlerLine2D(numpoints=1)})
plt.show()

analog2 = Arduino_Analog(ARDUINO,[0,1,4])
analog2.set_log_interval_ms(100)
analog2.start_log()

log2 = analog2.get_log()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(range(len(log2[0])), log2[0], 
                  'ro', label="X-axis of joystick")
line2, = plt.plot(range(len(log2[1])), log2[1], 
                  'bs', label="Y-axis of joystick")
line3, = plt.plot(range(len(log2[2])), log2[2], 
                  'g^', label="potentiometer")
plt.title('Arduino Analog Voltage Log')
plt.axis([0, len(log2[0]), 0.0, 3.3])
plt.legend(loc=4,bbox_to_anchor=(1, -0.3),
           ncol=2, borderaxespad=0.,
           handler_map={line1: HandlerLine2D(numpoints=1),
                        line2: HandlerLine2D(numpoints=1),
                        line3: HandlerLine2D(numpoints=1)})
plt.show()

