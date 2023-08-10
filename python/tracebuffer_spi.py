from pprint import pprint
from time import sleep
from pynq import PL
from pynq import Overlay
from pynq.drivers import Trace_Buffer
from pynq.iop import Pmod_OLED
from pynq.iop import PMODA
from pynq.iop import PMODB
from pynq.iop import ARDUINO

ol = Overlay("base.bit")
ol.download()

oled = Pmod_OLED(PMODB)

tr_buf = Trace_Buffer(PMODB,pins=[0,1,2,3],probes=['CS','MOSI','NC','CLK'],
                      protocol="spi",rate=20000000)

# Start the trace buffer
tr_buf.start()

# Write characters
oled.write("1 2 3 4 5 6")

# Stop the trace buffer
tr_buf.stop()

# Configuration for PMODB
start = 25000
stop = 35000

# Parsing and decoding
tr_buf.parse("spi_trace.csv",start,stop)
tr_buf.decode("spi_trace.pd",
              options=':wordsize=8:cpol=0:cpha=0')

tr_buf.display()



