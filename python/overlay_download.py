# Using base.bit located in pynq package
from pynq import Overlay
ol = Overlay("base.bit")

# Using the same bitstream, but with full path
from pynq import Overlay
ol = Overlay("/home/xilinx/pynq/bitstream/base.bit")

ol.download()
ol.bitstream.timestamp

from pynq import PL
PL.bitfile_name

PL.timestamp

ol.is_loaded()

import time
import matplotlib.pyplot as plt
from pynq import Overlay

ol1 = Overlay("base.bit")
length = 50
log1 = []
for i in range(length):
    start = time.time()
    ol1.download()
    end = time.time()
    # Record milliseconds
    log1.append((end-start)*1000)

# Draw the figure
get_ipython().magic('matplotlib inline')
plt.plot(range(length), log1, 'ro')
plt.title('Bitstream loading time (ms)')
plt.axis([0, length, 0, 1000])
plt.show()

del ol1



