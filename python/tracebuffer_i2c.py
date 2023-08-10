from pprint import pprint
from time import sleep
from pynq import PL
from pynq import Overlay
from pynq.drivers import Trace_Buffer
from pynq.iop import Pmod_TMP2
from pynq.iop import PMODA
from pynq.iop import PMODB
from pynq.iop import ARDUINO

ol = Overlay("base.bit")
ol.download()

tmp2 = Pmod_TMP2(PMODA)
tmp2.set_log_interval_ms(1)

tr_buf = Trace_Buffer(PMODA,pins=[2,3],probes=['SCL','SDA'],
                      protocol="i2c",rate=1000000)

# Start the trace buffer
tr_buf.start()

# Issue reads for 1 second
tmp2.start_log()
sleep(1)
tmp2_log = tmp2.get_log()

# Stop the trace buffer
tr_buf.stop()

# Set up samples
start = 500
stop = 3500

# Parsing and decoding samples
tr_buf.parse("i2c_trace.csv",start,stop)
tr_buf.decode("i2c_trace.pd")

tr_buf.display()



