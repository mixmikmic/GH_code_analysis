from pynq import Overlay
from pynq.iop import Pmod_OLED
from pynq.iop import PMODB

ol = Overlay("base.bit")
ol.download()
oled = Pmod_OLED(PMODB)

oled.write("Hello World")

oled.clear()

from pynq.iop import Pmod_ALS
from pynq.iop import PMODA

als = Pmod_ALS(PMODA)
als.read()

oled.write("Light value : " + str(als.read()))

import time
from pynq.iop import Pmod_ALS
from pynq.iop import PMODA

als = Pmod_ALS(PMODA)

als.set_log_interval_ms(100)
als.start_log()
time.sleep(1)
als.stop_log()
als.get_log()

